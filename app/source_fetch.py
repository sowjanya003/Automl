"""
app/source_fetch.py

Single shared place for "pull a file from wherever it lives (Databricks
Unity Catalog Volume, OneLake/Fabric lakehouse) and hand back raw bytes".

This consolidates logic that used to be copy-pasted across six locations
in main.py:
    - full_background_upload_process   (/upload_file_V)
    - build_ml_model_dep               (/build_ml_model_dep)
    - test_model_v                     (/test_model_v)
    - full_background_training_process (/build_ml_model_v background thread)
    - build_ml_model_v                 (/build_ml_model_v sync branch)
    - full_background_kpi_process      (/dataset_kpis)

Behavior is unchanged from the original inline code:
    - If a Databricks handler is configured AND the path starts with
      "/Volumes/", fetch from Databricks.
    - Otherwise, fetch from OneLake using the hardcoded "agenticBI" /
      "newagenticBI" workspace/lakehouse and the VERITAS_* service
      principal credentials.

An explicit `source_type` can be passed to skip the prefix-sniffing (useful
for callers that already know the source, and for future source types like
Snowflake that won't have a recognizable path prefix at all). When
`source_type` is omitted, the existing prefix-based auto-detection is used
so all six existing call sites keep working without any request-shape
changes on the frontend.
"""

import os
import logging
from io import BytesIO
from typing import Optional

from azure.identity import ClientSecretCredential
from azure.storage.filedatalake import DataLakeServiceClient

from app.databricks_handler import DatabricksHandler, DatabricksFileNotFoundError
from app.onelake_handler import OneLake_handler
from app.snowflake_handler import (
    SnowflakeHandler,
    SnowflakeFileNotFoundError,
    SnowflakeReferenceError,
    is_snowflake_reference,
)

logger = logging.getLogger(__name__)


class SourceFileNotFoundError(FileNotFoundError):
    """Raised when the requested path doesn't exist at the resolved source."""
    pass


class SourceCredentialsMissingError(RuntimeError):
    """Raised when the required credentials for a source aren't configured."""
    pass


# Hardcoded OneLake workspace/lakehouse, matching the existing inline code
# at every one of the six original call sites.
_ONELAKE_WORKSPACE_NAME = "agenticBI"
_ONELAKE_LAKEHOUSE_NAME = "newagenticBI"


def detect_source_type(file_path: str, databricks_handler: Optional[DatabricksHandler]) -> str:
    """
    Auto-detection rule used at every existing call site:
        - a configured Databricks handler + a path starting with
          "/Volumes/" means Databricks.
        - a path starting with "snowflake://stage/" OR "@" (native
          Snowflake stage syntax) means Snowflake, regardless of
          whether a Databricks handler is configured (the two are
          independent, unlike the Databricks check which is gated on
          the handler existing).
        - anything else falls back to OneLake.
    """
    if databricks_handler and file_path.strip().startswith("/Volumes/"):
        return "databricks"
    if is_snowflake_reference(file_path):
        return "snowflake"
    return "onelake"


def _fetch_from_databricks(databricks_handler: DatabricksHandler, file_path: str) -> bytes:
    try:
        content = databricks_handler.download_file(file_path)
    except DatabricksFileNotFoundError:
        raise SourceFileNotFoundError(f"File not found at path: {file_path}")
    logger.info(f"File fetched from Databricks: {file_path}")
    return content


def _fetch_from_snowflake(file_path: str) -> bytes:
    snowflake_account = os.getenv("SNOWFLAKE_ACCOUNT")
    snowflake_user = os.getenv("SNOWFLAKE_USER")
    snowflake_password = os.getenv("SNOWFLAKE_PASSWORD")
    snowflake_warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")

    if not all([snowflake_account, snowflake_user, snowflake_password, snowflake_warehouse]):
        raise SourceCredentialsMissingError(
            "Snowflake credentials incomplete - check SNOWFLAKE_ACCOUNT / "
            "SNOWFLAKE_USER / SNOWFLAKE_PASSWORD / SNOWFLAKE_WAREHOUSE"
        )
    try:
        handler = SnowflakeHandler()
        content = handler.download_file(file_path)
    except SnowflakeReferenceError as e:
        raise ValueError(str(e))
    except SnowflakeFileNotFoundError:
        raise SourceFileNotFoundError(f"File not found at path: {file_path}")
    logger.info(f"File fetched from Snowflake: {file_path}")
    return content


def _fetch_from_onelake(file_path: str) -> bytes:
    veritas_tenant_id = os.getenv("VERITAS_TENANT_ID")
    veritas_client_id = os.getenv("VERITAS_CLIENT_ID")
    veritas_client_secret = os.getenv("VERITAS_CLIENT_SECRET")

    if not all([veritas_tenant_id, veritas_client_id, veritas_client_secret]):
        raise SourceCredentialsMissingError(
            "Veritas credentials incomplete – check VERITAS_TENANT_ID / "
            "VERITAS_CLIENT_ID / VERITAS_CLIENT_SECRET"
        )

    veritas_credential = ClientSecretCredential(
        tenant_id=veritas_tenant_id,
        client_id=veritas_client_id,
        client_secret=veritas_client_secret,
    )

    veritas_handler = OneLake_handler()
    veritas_handler.credential = veritas_credential

    ws_id = veritas_handler.get_workspace_id_by_name(_ONELAKE_WORKSPACE_NAME)
    lh_id = veritas_handler.get_lakehouse_id_by_name(_ONELAKE_WORKSPACE_NAME, _ONELAKE_LAKEHOUSE_NAME)

    full_path = f"{lh_id}/{file_path.strip('/')}"

    service_client = DataLakeServiceClient(
        account_url="https://onelake.dfs.fabric.microsoft.com",
        credential=veritas_credential,
    )
    fs_client = service_client.get_file_system_client(ws_id)
    file_client = fs_client.get_file_client(full_path)

    if not file_client.exists():
        raise SourceFileNotFoundError(f"File not found at path: {full_path}")

    download = file_client.download_file()
    logger.info(f"File fetched from OneLake: {full_path}")
    return download.readall()


def fetch_source_file(
    file_path: str,
    databricks_handler: Optional[DatabricksHandler],
    source_type: Optional[str] = None,
) -> "tuple[BytesIO, str]":
    """
    Fetch a file from whichever source it lives in and return
    (content_as_BytesIO, resolved_source_type).

    Raises:
        SourceFileNotFoundError        - path doesn't exist at the source
        SourceCredentialsMissingError   - required source credentials aren't configured
        ValueError                      - unknown/unsupported source_type passed explicitly
    """
    resolved_source_type = source_type or detect_source_type(file_path, databricks_handler)

    if resolved_source_type == "databricks":
        if not databricks_handler:
            raise SourceCredentialsMissingError("Databricks is not configured on this server")
        raw_bytes = _fetch_from_databricks(databricks_handler, file_path)
    elif resolved_source_type == "onelake":
        raw_bytes = _fetch_from_onelake(file_path)
    elif resolved_source_type == "snowflake":
        raw_bytes = _fetch_from_snowflake(file_path)
    else:
        raise ValueError(f"Unsupported source_type: {resolved_source_type}")

    content = BytesIO(raw_bytes)
    content.seek(0)
    return content, resolved_source_type


def fetch_train_and_optional_test_file(
    file_path: str,
    test_file_path: Optional[str],
    databricks_handler: Optional[DatabricksHandler],
    source_type: Optional[str] = None,
) -> "tuple[BytesIO, Optional[BytesIO], str]":
    """
    Convenience wrapper for the common "training file + optional test file,
    both from the same source" pattern used by the build_ml_model endpoints.

    Returns (train_content, test_content_or_None, resolved_source_type).
    Both files are fetched from the same resolved source type (whichever
    the training file resolves to), matching the original inline behavior.
    """
    train_content, resolved_source_type = fetch_source_file(
        file_path=file_path,
        databricks_handler=databricks_handler,
        source_type=source_type,
    )

    test_content = None
    if test_file_path:
        test_content, _ = fetch_source_file(
            file_path=test_file_path,
            databricks_handler=databricks_handler,
            source_type=resolved_source_type,
        )

    return train_content, test_content, resolved_source_type


# ----------------------------------------------------------------------
# Write-back: mirror an output artifact (predictions, processed dataset,
# etc.) into the same Databricks Volume the source file was read from.
#
# Blob storage remains the mandatory, always-written copy for every
# source type (unchanged). This is an ADDITIONAL copy, written only when
# the original file came from Databricks, so the user's own Databricks
# workspace also has a record of what our app produced from their data.
# ----------------------------------------------------------------------

def derive_databricks_volume_root(databricks_file_path: str) -> str:
    """
    Given a full Unity Catalog Volume path like
        /Volumes/catalog/schema/volume/some/nested/file.csv
    return the volume root:
        /Volumes/catalog/schema/volume

    Kept as a general-purpose utility (e.g. for future features that need
    to resolve a Volume root from an arbitrary path), but NOTE: write-back
    of AutoML-generated outputs deliberately does NOT use this against the
    source file's own path anymore - see write_back_to_databricks() below,
    which writes to a separately configured output catalog instead.
    """
    parts = databricks_file_path.strip("/").split("/")
    # parts[0] == "Volumes", parts[1]=catalog, parts[2]=schema, parts[3]=volume
    if len(parts) < 4 or parts[0].lower() != "volumes":
        raise ValueError(
            f"Cannot derive volume root from '{databricks_file_path}': "
            "expected /Volumes/<catalog>/<schema>/<volume>/..."
        )
    return "/" + "/".join(parts[:4])


def get_databricks_output_root() -> Optional[str]:
    """
    Root Volume path for AutoML-generated outputs, read fresh from the
    environment each call (so a config change takes effect without needing
    to reimport this module). Returns None if not configured, in which
    case write-back is skipped entirely - Blob remains the sole copy.
    """
    root = os.getenv("DATABRICKS_OUTPUT_ROOT")
    if not root:
        return None
    return "/" + root.strip("/")


# Tracks which output roots we've already confirmed/created this process,
# so we don't hit the schema/volume-existence API on every single upload -
# only once per root, per process lifetime.
_provisioned_output_roots: set = set()


def _ensure_output_root_provisioned(databricks_handler: DatabricksHandler, output_root: str) -> None:
    """
    Best-effort, cached check that the schema+volume in output_root exist,
    creating them if not (assumes the catalog itself already exists).
    Never raises - if this fails (e.g. insufficient permissions), the
    subsequent upload attempt will simply fail too and get logged there,
    same as any other write-back failure.
    """
    if output_root in _provisioned_output_roots:
        return
    try:
        databricks_handler.ensure_output_path_exists(output_root)
        _provisioned_output_roots.add(output_root)
    except Exception as e:
        logger.warning(
            f"Could not verify/create Databricks schema+volume for "
            f"'{output_root}' (non-fatal, upload will be attempted anyway): {e}"
        )


def write_back_to_databricks(
    databricks_handler: Optional[DatabricksHandler],
    source_file_path: str,
    relative_suffix: str,
    content: bytes,
) -> Optional[str]:
    """
    Best-effort mirror of an AutoML-generated artifact (currently:
    predictions) into a dedicated Databricks output catalog - separate
    from wherever the source dataset was read from (e.g.
    veriton-db.landing.datasets), configured via DATABRICKS_OUTPUT_ROOT.

    Never raises - a failed or unconfigured write-back should not fail the
    caller's main flow, since Blob already has the authoritative copy.

    `source_file_path` is kept as a parameter (unused for path derivation
    now) so call sites don't need to change; it's retained for potential
    future use (e.g. logging/traceability of which source triggered this
    write-back).

    Returns the Databricks path written to, or None if skipped/failed.
    """
    if not databricks_handler:
        return None

    output_root = get_databricks_output_root()
    if not output_root:
        logger.info(
            "DATABRICKS_OUTPUT_ROOT not configured - skipping Databricks "
            "write-back, Blob copy remains the only stored copy."
        )
        return None

    _ensure_output_root_provisioned(databricks_handler, output_root)

    try:
        target_path = f"{output_root}/{relative_suffix.strip('/')}"
        databricks_handler.upload_file(target_path, content, overwrite=True)
        logger.info(f"Mirrored artifact to Databricks output catalog: {target_path}")
        return target_path
    except Exception as e:
        logger.warning(f"Databricks write-back skipped (non-fatal): {e}")
        return None


def get_onelake_output_root() -> Optional[str]:
    """
    Root path (relative to the fixed OneLake workspace/lakehouse's Files/
    area) for AutoML-generated outputs, read fresh from the environment
    each call. Returns None if not configured, in which case write-back
    is skipped entirely - Blob remains the sole copy.
    """
    root = os.getenv("ONELAKE_OUTPUT_ROOT", "Files/automl-outputs")
    if not root:
        return None
    return root.strip("/")


def get_snowflake_output_root() -> Optional[str]:
    """
    Root stage reference for AutoML-generated outputs, read fresh from
    the environment each call. Returns None if not configured, in which
    case write-back is skipped entirely - Blob remains the sole copy.
    """
    return os.getenv("SNOWFLAKE_OUTPUT_ROOT")


# Tracks which Snowflake output roots we've already confirmed/created
# this process, mirroring _provisioned_output_roots for Databricks.
_provisioned_snowflake_roots: set = set()


def _ensure_snowflake_output_root_provisioned(snowflake_handler: "SnowflakeHandler", output_root: str) -> None:
    """
    Best-effort, cached check that the schema+stage in output_root exist,
    creating them if not (assumes the database itself already exists).
    Never raises - if this fails (e.g. insufficient permissions), the
    subsequent upload attempt will simply fail too and get logged there,
    same as any other write-back failure.
    """
    if output_root in _provisioned_snowflake_roots:
        return
    try:
        snowflake_handler.ensure_output_stage_exists(output_root)
        _provisioned_snowflake_roots.add(output_root)
    except Exception as e:
        logger.warning(
            f"Could not verify/create Snowflake schema+stage for "
            f"'{output_root}' (non-fatal, upload will be attempted anyway): {e}"
        )


def write_back_to_snowflake(
    snowflake_handler: Optional["SnowflakeHandler"],
    source_file_path: str,
    relative_suffix: str,
    content: bytes,
) -> Optional[str]:
    """
    Best-effort mirror of an AutoML-generated artifact (predictions, run
    artifacts) into a dedicated Snowflake output stage - separate from
    wherever the source dataset was read from (e.g.
    VERITON_DB.DATASETS.DATASETS_STAGE), configured via
    SNOWFLAKE_OUTPUT_ROOT.

    Never raises - a failed or unconfigured write-back should not fail
    the caller's main flow, since Blob already has the authoritative copy.

    `source_file_path` is kept as a parameter (unused for path derivation)
    for symmetry with write_back_to_databricks() and potential future use.

    Returns the Snowflake reference written to, or None if skipped/failed.
    """
    if not snowflake_handler:
        return None

    output_root = get_snowflake_output_root()
    if not output_root:
        logger.info(
            "SNOWFLAKE_OUTPUT_ROOT not configured - skipping Snowflake "
            "write-back, Blob copy remains the only stored copy."
        )
        return None

    _ensure_snowflake_output_root_provisioned(snowflake_handler, output_root)

    try:
        target_path = f"{output_root}/{relative_suffix.strip('/')}"
        snowflake_handler.upload_file(target_path, content)
        logger.info(f"Mirrored artifact to Snowflake output stage: {target_path}")
        return target_path
    except Exception as e:
        logger.warning(f"Snowflake write-back skipped (non-fatal): {e}")
        return None


# ----------------------------------------------------------------------
# Write-back: mirror an output artifact (predictions, run artifacts, etc.)
# into the same OneLake lakehouse the source file was read from.
#
# Blob storage remains the mandatory, always-written copy for every
# source type (unchanged). This is an ADDITIONAL copy, written only when
# the original file came from OneLake, mirroring write_back_to_databricks
# / write_back_to_snowflake above. Deliberately reuses the exact same
# VERITAS_* credentials and hardcoded workspace/lakehouse
# (_ONELAKE_WORKSPACE_NAME / _ONELAKE_LAKEHOUSE_NAME) that
# _fetch_from_onelake() already uses for reads, so switching credentials
# later (e.g. from a test service principal to the real Veritas one) is a
# pure env-var change - no code change needed here.
# ----------------------------------------------------------------------

# Tracks which output roots we've already confirmed/created this process,
# mirroring _provisioned_output_roots / _provisioned_snowflake_roots
# above. Kept for symmetry even though OneLake/ADLS Gen2 auto-creates
# missing intermediate folders on upload, so no explicit provisioning
# call is actually required here.
_provisioned_onelake_output_roots: set = set()


def write_back_to_onelake(
    source_file_path: str,
    relative_suffix: str,
    content: bytes,
) -> Optional[str]:
    """
    Best-effort mirror of an AutoML-generated artifact (predictions, run
    artifacts) into a dedicated folder inside the same OneLake lakehouse
    used for reads - configured via ONELAKE_OUTPUT_ROOT (default
    "Files/automl-outputs").

    Never raises - a failed or unconfigured write-back should not fail
    the caller's main flow, since Blob already has the authoritative copy.

    `source_file_path` is kept as a parameter (unused for path derivation)
    for symmetry with write_back_to_databricks() / write_back_to_snowflake()
    and potential future use.

    Returns the OneLake path written to, or None if skipped/failed.
    """
    output_root = get_onelake_output_root()
    if not output_root:
        logger.info(
            "ONELAKE_OUTPUT_ROOT not configured - skipping OneLake "
            "write-back, Blob copy remains the only stored copy."
        )
        return None

    veritas_tenant_id = os.getenv("VERITAS_TENANT_ID")
    veritas_client_id = os.getenv("VERITAS_CLIENT_ID")
    veritas_client_secret = os.getenv("VERITAS_CLIENT_SECRET")

    if not all([veritas_tenant_id, veritas_client_id, veritas_client_secret]):
        logger.info(
            "VERITAS credentials not configured - skipping OneLake "
            "write-back, Blob copy remains the only stored copy."
        )
        return None

    try:
        veritas_credential = ClientSecretCredential(
            tenant_id=veritas_tenant_id,
            client_id=veritas_client_id,
            client_secret=veritas_client_secret,
        )

        veritas_handler = OneLake_handler()
        veritas_handler.credential = veritas_credential

        ws_id = veritas_handler.get_workspace_id_by_name(_ONELAKE_WORKSPACE_NAME)
        lh_id = veritas_handler.get_lakehouse_id_by_name(_ONELAKE_WORKSPACE_NAME, _ONELAKE_LAKEHOUSE_NAME)

        target_path = f"{output_root}/{relative_suffix.strip('/')}"
        written_path = veritas_handler.upload_file(
            workspace_id=ws_id,
            lakehouse_id=lh_id,
            relative_path=target_path,
            content=content,
            overwrite=True,
        )
        logger.info(f"Mirrored artifact to OneLake output folder: {written_path}")
        return target_path
    except Exception as e:
        logger.warning(f"OneLake write-back skipped (non-fatal): {e}")
        return None