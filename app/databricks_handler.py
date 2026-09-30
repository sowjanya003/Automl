"""
Databricks handler for fetching files from Unity Catalog Volumes.

Replaces the OneLake/Fabric fetch pattern used elsewhere in this codebase.
Uses the Databricks Files API (REST) authenticated with a Personal Access
Token — no cluster is required to read/write Volume files, so
DATABRICKS_CLUSTER_ID is accepted for future use (e.g. running notebooks/
SQL) but is not needed for the download/upload/list operations here.

Docs: https://docs.databricks.com/api/workspace/files

Unity Catalog Volume paths look like:
    /Volumes/<catalog>/<schema>/<volume>/<path...>/<file>
e.g.
    /Volumes/veriton-db/landing/datasets/user123/job1/orders.csv
"""

import os
import logging
from typing import Optional, List, Dict, Any

import requests

logger = logging.getLogger(__name__)


class DatabricksFileNotFoundError(FileNotFoundError):
    """Raised when a requested path does not exist in the Volume."""
    pass


class DatabricksHandler:
    def __init__(
        self,
        host: Optional[str] = None,
        token: Optional[str] = None,
        cluster_id: Optional[str] = None,
    ):
        self.host = (host or os.getenv("DATABRICKS_HOST", "")).rstrip("/")
        self.token = token or os.getenv("DATABRICKS_TOKEN")
        self.cluster_id = cluster_id or os.getenv("DATABRICKS_CLUSTER_ID")

        if not self.host or not self.token:
            raise ValueError(
                "Missing Databricks credentials: set DATABRICKS_HOST and DATABRICKS_TOKEN"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    @staticmethod
    def _normalize_volume_path(path: str) -> str:
        """
        Accepts a Unity Catalog Volume path and returns a clean, absolute
        form starting with /Volumes/. Tolerates missing/extra slashes and
        case variants of the 'Volumes' segment.
        """
        if not path:
            raise ValueError("Path is required, e.g. /Volumes/catalog/schema/volume/file.csv")

        clean = "/" + path.strip().strip("/")

        if clean.lower().startswith("/volumes/"):
            clean = "/Volumes/" + clean[len("/volumes/"):]
        else:
            raise ValueError(
                f"Unsupported Databricks path '{path}'. Expected a Unity Catalog "
                "Volume path like /Volumes/<catalog>/<schema>/<volume>/<file>"
            )
        return clean

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def file_exists(self, volume_path: str) -> bool:
        clean = self._normalize_volume_path(volume_path)
        url = f"{self.host}/api/2.0/fs/files{clean}"
        r = requests.head(url, headers=self._headers(), timeout=30)
        if r.status_code == 200:
            return True
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return False

    def download_file(self, volume_path: str) -> bytes:
        """
        Download the raw bytes of a file stored in a Unity Catalog Volume.
        Raises DatabricksFileNotFoundError if the path doesn't exist.
        """
        clean = self._normalize_volume_path(volume_path)
        url = f"{self.host}/api/2.0/fs/files{clean}"

        r = requests.get(url, headers=self._headers(), timeout=120)
        if r.status_code == 404:
            raise DatabricksFileNotFoundError(f"File not found in Databricks Volume: {clean}")
        r.raise_for_status()
        return r.content

    def upload_file(self, volume_path: str, content: bytes, overwrite: bool = True) -> None:
        """Upload raw bytes to a Unity Catalog Volume path."""
        clean = self._normalize_volume_path(volume_path)
        url = f"{self.host}/api/2.0/fs/files{clean}"
        params = {"overwrite": str(overwrite).lower()}

        r = requests.put(url, headers=self._headers(), params=params, data=content, timeout=120)
        r.raise_for_status()

    def list_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """List contents of a Unity Catalog Volume directory."""
        clean = self._normalize_volume_path(dir_path)
        url = f"{self.host}/api/2.0/fs/directories{clean}"

        r = requests.get(url, headers=self._headers(), timeout=60)
        if r.status_code == 404:
            raise DatabricksFileNotFoundError(f"Directory not found in Databricks Volume: {clean}")
        r.raise_for_status()
        return r.json().get("contents", [])

    # ------------------------------------------------------------------
    # Schema / Volume auto-provisioning
    #
    # Assumes the CATALOG already exists (catalog creation is a governance
    # decision left to a Databricks admin, done once manually). Schema and
    # Volume are lower-stakes, scoped inside an already-approved catalog,
    # so the app can create them on first use if missing.
    # ------------------------------------------------------------------
    def schema_exists(self, catalog: str, schema: str) -> bool:
        url = f"{self.host}/api/2.1/unity-catalog/schemas/{catalog}.{schema}"
        r = requests.get(url, headers=self._headers(), timeout=30)
        if r.status_code == 200:
            return True
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return False

    def create_schema(self, catalog: str, schema: str) -> None:
        url = f"{self.host}/api/2.1/unity-catalog/schemas"
        body = {"name": schema, "catalog_name": catalog}
        r = requests.post(url, headers=self._headers(), json=body, timeout=30)
        # 409 = already exists (race with another process) - treat as success
        if r.status_code not in (200, 409):
            r.raise_for_status()

    def volume_exists(self, catalog: str, schema: str, volume: str) -> bool:
        url = f"{self.host}/api/2.1/unity-catalog/volumes/{catalog}.{schema}.{volume}"
        r = requests.get(url, headers=self._headers(), timeout=30)
        if r.status_code == 200:
            return True
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return False

    def create_volume(self, catalog: str, schema: str, volume: str) -> None:
        url = f"{self.host}/api/2.1/unity-catalog/volumes"
        body = {
            "name": volume,
            "catalog_name": catalog,
            "schema_name": schema,
            "volume_type": "MANAGED",
        }
        r = requests.post(url, headers=self._headers(), json=body, timeout=30)
        if r.status_code not in (200, 409):
            r.raise_for_status()

    def ensure_output_path_exists(self, volume_root_path: str) -> None:
        """
        Given a Volume root like /Volumes/<catalog>/<schema>/<volume>,
        create the schema and/or volume if they don't already exist.
        Assumes <catalog> already exists. Raises if creation fails for a
        reason other than "already exists" (e.g. insufficient permissions) -
        callers should catch and treat as non-fatal, same as any other
        write-back failure.
        """
        clean = self._normalize_volume_path(volume_root_path)
        parts = clean.strip("/").split("/")
        if len(parts) < 4:
            raise ValueError(
                f"Expected /Volumes/<catalog>/<schema>/<volume>, got '{volume_root_path}'"
            )
        catalog, schema, volume = parts[1], parts[2], parts[3]

        if not self.schema_exists(catalog, schema):
            logger.info(f"Creating missing Databricks schema: {catalog}.{schema}")
            self.create_schema(catalog, schema)

        if not self.volume_exists(catalog, schema, volume):
            logger.info(f"Creating missing Databricks volume: {catalog}.{schema}.{volume}")
            self.create_volume(catalog, schema, volume)


_databricks_handler_singleton: Optional[DatabricksHandler] = None


def get_databricks_handler() -> Optional[DatabricksHandler]:
    """
    Lazily builds a module-level DatabricksHandler singleton from env vars.
    Returns None (rather than raising) if credentials aren't configured, so
    callers can return a clean 503 instead of crashing at import time.
    """
    global _databricks_handler_singleton
    if _databricks_handler_singleton is not None:
        return _databricks_handler_singleton

    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    cluster_id = os.getenv("DATABRICKS_CLUSTER_ID")

    if not host or not token:
        logger.warning("Databricks credentials not configured (DATABRICKS_HOST / DATABRICKS_TOKEN missing)")
        return None

    _databricks_handler_singleton = DatabricksHandler(host=host, token=token, cluster_id=cluster_id)
    return _databricks_handler_singleton