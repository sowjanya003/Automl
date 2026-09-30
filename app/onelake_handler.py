
import os
from fastapi import FastAPI, Path, Query, HTTPException
from typing import List, Dict, Any
import requests
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential

class OneLake_handler:
    def __init__(self):
        self.credential = DefaultAzureCredential()   # still needed for OneLake (storage)
        
    def _auth_header(self):
        token = self.credential.get_token(
            "https://api.fabric.microsoft.com/.default"
        ).token
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def _get(self, url: str) -> dict:
        r = requests.get(url, headers=self._auth_header())
        r.raise_for_status()
        return r.json()

    def get_workspaces(self) -> List[Dict[str, Any]]:
        full_data = self._get("https://api.fabric.microsoft.com/v1/workspaces").get("value", [])
        return [{"id": ws["id"], "name": ws["displayName"]} for ws in full_data]

    def get_workspace_id_by_name(self, workspace_name: str) -> str:
        workspaces = self.get_workspaces()
        for ws in workspaces:
            if ws["name"].lower() == workspace_name.lower():  # Case-insensitive match
                return ws["id"]
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_name}' not found")

    def get_lakehouses(self, workspace_id: str) -> List[Dict[str, Any]]:
        full_data = self._get(f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}/lakehouses").get("value", [])
        return [{"id": lh["id"], "name": lh["displayName"]} for lh in full_data]

    def get_lakehouse_id_by_name(self, workspace_name: str, lakehouse_name: str) -> str:
        workspace_id = self.get_workspace_id_by_name(workspace_name)
        lakehouses = self.get_lakehouses(workspace_id)
        for lh in lakehouses:
            if lh["name"].lower() == lakehouse_name.lower():  # Case-insensitive match
                return lh["id"]
        raise HTTPException(status_code=404, detail=f"Lakehouse '{lakehouse_name}' not found in workspace '{workspace_name}'")

    def get_tables(self, workspace_id: str, lakehouse_id: str) -> List[Dict[str, Any]]:
        return self._get(f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/tables").get("value", [])

    def list_folder_contents(self, workspace_id: str, lakehouse_id: str, path: str = "Files") -> List[Dict[str, Any]]:
        service_client = DataLakeServiceClient(
            account_url="https://onelake.dfs.fabric.microsoft.com",
            credential=self.credential
        )
        fs_client = service_client.get_file_system_client(workspace_id)  # Filesystem = workspace GUID

        # Verify filesystem
        try:
            fs_client.get_file_system_properties()
        except Exception as e:
            raise ValueError(f"Filesystem '{workspace_id}' not found: {e}")

        # Path prefix: lakehouse_id/Files/{path}
        prefix = f"{lakehouse_id}/Files/" if path == "Files" else f"{lakehouse_id}/Files/{path.strip('/')}/"

        items = []
        try:
            paths = fs_client.get_paths(path=prefix, recursive=False)
            for p in paths:
                relative_path = p.name.replace(prefix.rstrip('/'), '').lstrip('/')
                name = relative_path.split("/")[-1] if "/" in relative_path else relative_path
                items.append({
                    "name": name,
                    "full_path": p.name,
                    "is_directory": p.is_directory,
                    "size": p.content_length if not p.is_directory else None,
                    "last_modified": p.last_modified.isoformat() if p.last_modified else None
                })
        except Exception as e:
            raise ValueError(f"Error listing '{prefix}': {e}")

        return items

    def upload_file(
        self,
        workspace_id: str,
        lakehouse_id: str,
        relative_path: str,
        content: bytes,
        overwrite: bool = True,
    ) -> str:
        """
        Write-back: upload raw bytes to a path inside this lakehouse's
        Files/ area. `relative_path` is anything under the lakehouse root,
        e.g. "Files/automl-outputs/<user_id>/runs/<run_id>/model.joblib".

        ADLS Gen2 auto-creates any missing intermediate folders on file
        creation, so no separate provisioning step is required (unlike the
        Databricks schema+volume / Snowflake schema+stage checks).

        Returns the full ADLS path written to.
        """
        service_client = DataLakeServiceClient(
            account_url="https://onelake.dfs.fabric.microsoft.com",
            credential=self.credential
        )
        fs_client = service_client.get_file_system_client(workspace_id)

        clean_path = relative_path.strip("/")
        full_path = f"{lakehouse_id}/{clean_path}"

        file_client = fs_client.get_file_client(full_path)
        file_client.upload_data(content, overwrite=overwrite)

        return full_path