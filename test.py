# # """
# # Download a single file from Snowflake by just giving its path.

# # Works for BOTH reference styles:
# #     Table:      snowflake://<database>/<schema>/<table>
# #     Stage file: snowflake://stage/<database>/<schema>/<stage>/<path...>

# # Usage:
# #     python download_by_path.py "snowflake://stage/VERITON_DB/DATASETS/DATASETS_STAGE/01/01/USR_01_JOB_01_dataset.csv"

# # Saves the downloaded bytes next to this script (or to --output if given)
# # and prints a preview.
# # """

# # import os
# # import sys
# # import argparse

# # sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# # from dotenv import load_dotenv
# # load_dotenv()

# # from app.snowflake_handler import (
# #     SnowflakeHandler,
# #     SnowflakeTableNotFoundError,
# #     SnowflakeReferenceError,
# # )
# # from app.snowflake_handler import SnowflakeHandler
# # print("Loaded from:", SnowflakeHandler.__module__, __import__("app.snowflake_handler", fromlist=["x"]).__file__)

# # def main():
# #     parser = argparse.ArgumentParser(description="Download a file from Snowflake by path")
# #     parser.add_argument("path", help="snowflake:// reference (table or stage file)")
# #     parser.add_argument("--output", help="Where to save the downloaded bytes (default: derived from filename)")
# #     args = parser.parse_args()

# #     account = os.getenv("SNOWFLAKE_ACCOUNT")
# #     user = os.getenv("SNOWFLAKE_USER")
# #     password = os.getenv("SNOWFLAKE_PASSWORD")
# #     warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
# #     role = os.getenv("SNOWFLAKE_ROLE")

# #     if not all([account, user, password, warehouse]):
# #         print(
# #             "FAIL: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, "
# #             "and/or SNOWFLAKE_WAREHOUSE not set in your environment/.env"
# #         )
# #         sys.exit(1)

# #     print(f"Path: {args.path}")
# #     print("-" * 60)

# #     handler = SnowflakeHandler(
# #         account=account, user=user, password=password,
# #         warehouse=warehouse, role=role
# #     )

# #     try:
# #         content = handler.fetch_by_reference(args.path)
# #     except SnowflakeReferenceError as e:
# #         print(f"[FAIL] Bad path: {e}")
# #         sys.exit(1)
# #     except SnowflakeTableNotFoundError as e:
# #         print(f"[FAIL] Not found: {e}")
# #         sys.exit(1)
# #     except Exception as e:
# #         print(f"[FAIL] {type(e).__name__}: {e}")
# #         sys.exit(1)

# #     print(f"[PASS] Downloaded {len(content)} bytes")

# #     output_path = args.output or ("downloaded_" + args.path.rstrip("/").split("/")[-1])
# #     with open(output_path, "wb") as f:
# #         f.write(content)
# #     print(f"[PASS] Saved to: {output_path}")

# #     preview = content[:300]
# #     try:
# #         print(f"\nPreview:\n{preview.decode('utf-8', errors='replace')}")
# #     except Exception:
# #         print(f"\nPreview (binary): {preview!r}")


# # if __name__ == "__main__":
# #     main()


# # # """
# # # Lists the Snowflake databases (and optionally schemas/tables) visible to
# # # your configured user. Useful for finding real values to plug into
# # # snowflake:// references before running test_snowflake_live.py.

# # # Usage:
# # #     # List all databases the user can see
# # #     python list_snowflake_databases.py

# # #     # List schemas inside a specific database
# # #     python list_snowflake_databases.py --database ANALYTICS_DB

# # #     # List tables inside a specific database.schema
# # #     python list_snowflake_databases.py --database ANALYTICS_DB --schema PUBLIC
# # # """

# # # import os
# # # import sys
# # # import argparse

# # # sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# # # from dotenv import load_dotenv
# # # load_dotenv()

# # # import snowflake.connector


# # # def get_connection():
# # #     account = os.getenv("SNOWFLAKE_ACCOUNT")
# # #     user = os.getenv("SNOWFLAKE_USER")
# # #     password = os.getenv("SNOWFLAKE_PASSWORD")
# # #     warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
# # #     role = os.getenv("SNOWFLAKE_ROLE")

# # #     if not all([account, user, password, warehouse]):
# # #         print(
# # #             "FAIL: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, "
# # #             "and/or SNOWFLAKE_WAREHOUSE not set in your environment/.env"
# # #         )
# # #         sys.exit(1)

# # #     kwargs = dict(account=account, user=user, password=password, warehouse=warehouse)
# # #     if role:
# # #         kwargs["role"] = role

# # #     print(f"Connecting to Snowflake account={account}, user={user}, warehouse={warehouse}, "
# # #           f"role={role or '(default)'} ...")
# # #     return snowflake.connector.connect(**kwargs)


# # # def list_databases(cur):
# # #     cur.execute("SHOW DATABASES")
# # #     rows = cur.fetchall()
# # #     columns = [d[0] for d in cur.description]
# # #     name_idx = columns.index("name") if "name" in columns else 1

# # #     print(f"\nDatabases visible to this user ({len(rows)} found):")
# # #     print("-" * 60)
# # #     for row in rows:
# # #         print(f"  - {row[name_idx]}")


# # # def list_schemas(cur, database):
# # #     cur.execute(f"SHOW SCHEMAS IN DATABASE {database}")
# # #     rows = cur.fetchall()
# # #     columns = [d[0] for d in cur.description]
# # #     name_idx = columns.index("name") if "name" in columns else 1

# # #     print(f"\nSchemas in {database} ({len(rows)} found):")
# # #     print("-" * 60)
# # #     for row in rows:
# # #         print(f"  - {database}.{row[name_idx]}")


# # # def list_tables(cur, database, schema):
# # #     cur.execute(f"SHOW TABLES IN SCHEMA {database}.{schema}")
# # #     rows = cur.fetchall()
# # #     columns = [d[0] for d in cur.description]
# # #     name_idx = columns.index("name") if "name" in columns else 1

# # #     print(f"\nTables in {database}.{schema} ({len(rows)} found):")
# # #     print("-" * 60)
# # #     if not rows:
# # #         print("  (none found - this schema may be empty, or contain only views)")
# # #     for row in rows:
# # #         table_name = row[name_idx]
# # #         print(f"  - {database}.{schema}.{table_name}")
# # #         print(f"      snowflake reference: snowflake://{database}/{schema}/{table_name}")


# # # def main():
# # #     parser = argparse.ArgumentParser(description="List accessible Snowflake databases/schemas/tables")
# # #     parser.add_argument("--database", help="Database to drill into (lists its schemas)")
# # #     parser.add_argument("--schema", help="Schema to drill into (requires --database; lists its tables)")
# # #     args = parser.parse_args()

# # #     if args.schema and not args.database:
# # #         print("FAIL: --schema requires --database to also be given")
# # #         sys.exit(1)

# # #     conn = get_connection()
# # #     try:
# # #         cur = conn.cursor()
# # #         try:
# # #             if args.database and args.schema:
# # #                 list_tables(cur, args.database, args.schema)
# # #             elif args.database:
# # #                 list_schemas(cur, args.database)
# # #             else:
# # #                 list_databases(cur)
# # #         finally:
# # #             cur.close()
# # #     except Exception as e:
# # #         print(f"\nFAIL: {e}")
# # #         sys.exit(1)
# # #     finally:
# # #         conn.close()

# # #     print("\nDone.")


# # # if __name__ == "__main__":
# # #     main()

# # # # """
# # # # Recursively explore a Databricks Unity Catalog Volume and report:
# # # #   - total number of folders (including the root and all nested subfolders)
# # # #   - total number of files
# # # #   - a per-folder breakdown: how many files/subfolders each folder contains

# # # # Usage:
# # # #     python tests/explore_databricks_volume.py "/Volumes/veriton-db/landing/datasets"

# # # # If no path is given, falls back to DATABRICKS_TEST_FILE_PATH's parent dir,
# # # # or the sample root baked in below.

# # # # Notes:
# # # #   - Uses only DatabricksHandler.list_directory(), which calls the
# # # #     Databricks Files API's /api/2.0/fs/directories endpoint (non-recursive
# # # #     per call) — this script does the recursion itself, one folder at a time.
# # # #   - Safe for a few hundred to a few thousand folders. For very large
# # # #     volumes, add --max-depth to limit how deep it walks.
# # # # """

# # # # import os
# # # # import sys
# # # # import argparse

# # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # from dotenv import load_dotenv
# # # # load_dotenv()

# # # # from app.databricks_handler import DatabricksHandler, DatabricksFileNotFoundError

# # # # DEFAULT_ROOT = "/Volumes/veriton-db/landing/datasets"


# # # # def entry_name(entry: dict, parent_path: str) -> str:
# # # #     """Derive a display name from a directory entry, regardless of whether
# # # #     the API returned a 'name' field or only a full 'path'."""
# # # #     if entry.get("name"):
# # # #         return entry["name"]
# # # #     path = entry.get("path", "")
# # # #     return path.rstrip("/").rsplit("/", 1)[-1] if path else "(unknown)"


# # # # def entry_path(entry: dict, parent_path: str, name: str) -> str:
# # # #     """Get the full path for a directory entry, building it if not provided."""
# # # #     if entry.get("path"):
# # # #         return entry["path"]
# # # #     return f"{parent_path.rstrip('/')}/{name}"


# # # # def walk(handler: DatabricksHandler, root: str, max_depth: int = None):
# # # #     """
# # # #     Yields (folder_path, subfolder_names, file_names) for root and every
# # # #     nested folder, similar in spirit to os.walk().
# # # #     """
# # # #     stack = [(root, 0)]
# # # #     while stack:
# # # #         current_path, depth = stack.pop()

# # # #         if max_depth is not None and depth > max_depth:
# # # #             continue

# # # #         try:
# # # #             items = handler.list_directory(current_path)
# # # #         except DatabricksFileNotFoundError:
# # # #             print(f"[SKIP] Not found (may have been removed mid-walk): {current_path}")
# # # #             continue
# # # #         except Exception as e:
# # # #             print(f"[SKIP] Error listing {current_path}: {e}")
# # # #             continue

# # # #         subfolders = []
# # # #         files = []
# # # #         for item in items:
# # # #             name = entry_name(item, current_path)
# # # #             full_path = entry_path(item, current_path, name)
# # # #             if item.get("is_directory"):
# # # #                 subfolders.append(name)
# # # #                 stack.append((full_path, depth + 1))
# # # #             else:
# # # #                 files.append(name)

# # # #         yield current_path, subfolders, files


# # # # def main():
# # # #     parser = argparse.ArgumentParser(description=__doc__)
# # # #     parser.add_argument("root", nargs="?", default=None, help="Root Volume path to explore")
# # # #     parser.add_argument("--max-depth", type=int, default=None, help="Limit recursion depth")
# # # #     args = parser.parse_args()

# # # #     host = os.getenv("DATABRICKS_HOST")
# # # #     token = os.getenv("DATABRICKS_TOKEN")
# # # #     cluster_id = os.getenv("DATABRICKS_CLUSTER_ID")

# # # #     if not host or not token:
# # # #         print("FAIL: DATABRICKS_HOST and/or DATABRICKS_TOKEN not set in your environment/.env")
# # # #         sys.exit(1)

# # # #     root = args.root or os.getenv("DATABRICKS_EXPLORE_ROOT", DEFAULT_ROOT)

# # # #     print(f"Host:   {host}")
# # # #     print(f"Root:   {root}")
# # # #     print("-" * 70)

# # # #     handler = DatabricksHandler(host=host, token=token, cluster_id=cluster_id)

# # # #     total_folders = 0
# # # #     total_files = 0
# # # #     per_folder_summary = []

# # # #     for folder_path, subfolders, files in walk(handler, root, max_depth=args.max_depth):
# # # #         total_folders += 1
# # # #         total_files += len(files)
# # # #         per_folder_summary.append((folder_path, len(subfolders), len(files)))

# # # #         print(f"[FOLDER] {folder_path}")
# # # #         print(f"         subfolders: {len(subfolders)}   files: {len(files)}")
# # # #         for f in files:
# # # #             print(f"           - {f}")

# # # #     print("-" * 70)
# # # #     print("SUMMARY")
# # # #     print(f"  Total folders (incl. root): {total_folders}")
# # # #     print(f"  Total files (all folders):  {total_files}")
# # # #     print("-" * 70)
# # # #     print(f"{'Folder':<60} {'Subfolders':>10} {'Files':>8}")
# # # #     for folder_path, n_sub, n_files in per_folder_summary:
# # # #         print(f"{folder_path:<60} {n_sub:>10} {n_files:>8}")


# # # # if __name__ == "__main__":
# # # #     main()





# # # # # """
# # # # # Manual live smoke test for the Databricks Volume integration.

# # # # # This hits your REAL Databricks workspace using DATABRICKS_HOST /
# # # # # DATABRICKS_TOKEN from your .env — no mocking. Use it once to confirm the
# # # # # credentials and file path actually work end-to-end before relying on the
# # # # # app's /build_ml_model_v, /test_model_v, /dataset_kpis, etc.

# # # # # Usage:
# # # # #     python tests/test_databricks_live.py "/Volumes/veriton-db/landing/datasets/user123/job1/orders.csv"

# # # # # If no path is given, it falls back to the DATABRICKS_TEST_FILE_PATH env
# # # # # var, or the sample path baked in below.
# # # # # """

# # # # # import os
# # # # # import sys

# # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # from dotenv import load_dotenv
# # # # # load_dotenv()

# # # # # from app.databricks_handler import DatabricksHandler, DatabricksFileNotFoundError

# # # # # DEFAULT_TEST_PATH = "/Volumes/veriton-db/landing/datasets/user123/job1/orders.csv"


# # # # # def main():
# # # # #     host = os.getenv("DATABRICKS_HOST")
# # # # #     token = os.getenv("DATABRICKS_TOKEN")
# # # # #     cluster_id = os.getenv("DATABRICKS_CLUSTER_ID")

# # # # #     if not host or not token:
# # # # #         print("FAIL: DATABRICKS_HOST and/or DATABRICKS_TOKEN not set in your environment/.env")
# # # # #         sys.exit(1)

# # # # #     file_path = (
# # # # #         sys.argv[1] if len(sys.argv) > 1
# # # # #         else os.getenv("DATABRICKS_TEST_FILE_PATH", DEFAULT_TEST_PATH)
# # # # #     )

# # # # #     print(f"Host:        {host}")
# # # # #     print(f"Cluster ID:  {cluster_id or '(not set — not required for file access)'}")
# # # # #     print(f"Test path:   {file_path}")
# # # # #     print("-" * 60)

# # # # #     handler = DatabricksHandler(host=host, token=token, cluster_id=cluster_id)

# # # # #     # 1. Path normalization
# # # # #     try:
# # # # #         normalized = handler._normalize_volume_path(file_path)
# # # # #         print(f"[PASS] Path normalization -> {normalized}")
# # # # #     except ValueError as e:
# # # # #         print(f"[FAIL] Path normalization: {e}")
# # # # #         sys.exit(1)

# # # # #     # 2. file_exists
# # # # #     try:
# # # # #         exists = handler.file_exists(file_path)
# # # # #         print(f"[{'PASS' if exists else 'WARN'}] file_exists() -> {exists}")
# # # # #     except Exception as e:
# # # # #         print(f"[FAIL] file_exists() raised: {e}")

# # # # #     # 3. download_file
# # # # #     try:
# # # # #         content = handler.download_file(file_path)
# # # # #         print(f"[PASS] download_file() -> {len(content)} bytes")
# # # # #         preview = content[:200]
# # # # #         print(f"       preview: {preview!r}")
# # # # #     except DatabricksFileNotFoundError as e:
# # # # #         print(f"[FAIL] File not found: {e}")
# # # # #         sys.exit(1)
# # # # #     except Exception as e:
# # # # #         print(f"[FAIL] download_file() raised: {e}")
# # # # #         sys.exit(1)

# # # # #     # 4. list_directory (parent folder of the test file)
# # # # #     parent_dir = file_path.rsplit("/", 1)[0]
# # # # #     try:
# # # # #         items = handler.list_directory(parent_dir)
# # # # #         print(f"[PASS] list_directory('{parent_dir}') -> {len(items)} entries")
# # # # #         for item in items[:10]:
# # # # #             print(f"       - {item.get('name')} (dir={item.get('is_directory')})")
# # # # #     except Exception as e:
# # # # #         print(f"[WARN] list_directory() raised: {e}")

# # # # #     print("-" * 60)
# # # # #     print("Done. If all steps show [PASS], the Databricks Volume integration is working.")


# # # # # if __name__ == "__main__":
# # # # #     main()





# # # # # # """
# # # # # # generate_maintenance_test_data.py
# # # # # # ---------------------------------

# # # # # # Generates a realistic maintenance work-order dataset for testing the
# # # # # # multi-dimensional KPIs (cost/downtime by plant x category x month).

# # # # # # The column names match what the KPI engine auto-detects, so uploading the
# # # # # # output produces a fully-populated `kpis.multi_dimensional` block and
# # # # # # `data_readiness` all-true.

# # # # # # Usage:
# # # # # #     python generate_maintenance_test_data.py
# # # # # #     python generate_maintenance_test_data.py --rows 10000 --out my_data.csv
# # # # # #     python generate_maintenance_test_data.py --plants 8 --months 12

# # # # # # Requires: pandas, numpy
# # # # # # """

# # # # # # import argparse
# # # # # # import numpy as np
# # # # # # import pandas as pd


# # # # # # def generate(
# # # # # #     n_rows: int = 6000,
# # # # # #     n_plants: int = 6,
# # # # # #     n_months: int = 6,
# # # # # #     n_machines: int = 150,
# # # # # #     seed: int = 2026,
# # # # # # ) -> pd.DataFrame:
# # # # # #     """Build a maintenance dataset with realistic per-plant / per-category variation."""
# # # # # #     rng = np.random.default_rng(seed)

# # # # # #     # ---- Dimensions ----
# # # # # #     # Plant names: split across two example sites for flavour.
# # # # # #     half = max(1, n_plants // 2)
# # # # # #     plants = (
# # # # # #         [f"Hyderabad-{i+1}" for i in range(half)]
# # # # # #         + [f"Vizag-{i+1}" for i in range(n_plants - half)]
# # # # # #     )

# # # # # #     categories = ["Breakdown", "Preventive", "Predictive", "Calibration", "Routine"]
# # # # # #     cat_probs = [0.30, 0.25, 0.15, 0.10, 0.20]

# # # # # #     machines = [f"EQ{i:04d}" for i in range(n_machines)]

# # # # # #     # Per-plant cost multiplier so plants differ (more useful in charts).
# # # # # #     plant_cost_factor = {
# # # # # #         p: round(rng.uniform(0.8, 1.35), 2) for p in plants
# # # # # #     }
# # # # # #     # Breakdown work is far more disruptive than routine maintenance.
# # # # # #     cat_downtime_factor = {
# # # # # #         "Breakdown": 3.0, "Preventive": 0.6, "Predictive": 0.4,
# # # # # #         "Calibration": 0.3, "Routine": 0.8,
# # # # # #     }

# # # # # #     # ---- Sample rows ----
# # # # # #     plant = rng.choice(plants, n_rows)
# # # # # #     category = rng.choice(categories, n_rows, p=cat_probs)

# # # # # #     # Dates spread evenly across the requested number of months, from Jan 2026.
# # # # # #     span_days = n_months * 30
# # # # # #     dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(
# # # # # #         rng.integers(0, span_days, n_rows), unit="D"
# # # # # #     )

# # # # # #     # Costs
# # # # # #     material = rng.uniform(500, 40000, n_rows)
# # # # # #     material *= np.array([plant_cost_factor[p] for p in plant])
# # # # # #     labour = rng.uniform(200, 15000, n_rows)
# # # # # #     external = np.where(
# # # # # #         category == "Breakdown",
# # # # # #         rng.uniform(5000, 30000, n_rows),   # breakdowns pull in outside services
# # # # # #         rng.uniform(0, 8000, n_rows),
# # # # # #     )

# # # # # #     # Downtime: exponential, scaled by category
# # # # # #     downtime = np.array(
# # # # # #         [rng.exponential(4 * cat_downtime_factor[c]) for c in category]
# # # # # #     ).round(1)

# # # # # #     df = pd.DataFrame({
# # # # # #         "work_order_id":   [f"WO{i:07d}" for i in range(n_rows)],
# # # # # #         "plant_id":        plant,
# # # # # #         "work_order_type": category,
# # # # # #         "equipment_id":    rng.choice(machines, n_rows),
# # # # # #         "notification_date": dates.strftime("%Y-%m-%d"),
# # # # # #         "material_cost_inr":         material.round(2),
# # # # # #         "labour_cost_inr":           labour.round(2),
# # # # # #         "external_service_cost_inr": external.round(2),
# # # # # #         "downtime_hours":            downtime,
# # # # # #         "planned_quantity":          rng.integers(100, 1000, n_rows),
# # # # # #         "produced_quantity":         rng.integers(80, 950, n_rows),
# # # # # #         "breakdown_flag":            (category == "Breakdown").astype(int),
# # # # # #     })
# # # # # #     df["actual_cost_inr"] = (
# # # # # #         df.material_cost_inr + df.labour_cost_inr + df.external_service_cost_inr
# # # # # #     ).round(2)

# # # # # #     return df


# # # # # # def main():
# # # # # #     parser = argparse.ArgumentParser(
# # # # # #         description="Generate maintenance test data for multi-dimensional KPIs."
# # # # # #     )
# # # # # #     parser.add_argument("--rows", type=int, default=6000, help="number of work orders")
# # # # # #     parser.add_argument("--plants", type=int, default=6, help="number of plants")
# # # # # #     parser.add_argument("--months", type=int, default=6, help="months of history")
# # # # # #     parser.add_argument("--machines", type=int, default=150, help="distinct machines")
# # # # # #     parser.add_argument("--seed", type=int, default=2026, help="random seed")
# # # # # #     parser.add_argument(
# # # # # #         "--out", type=str, default="maintenance_multidim_test.csv",
# # # # # #         help="output CSV path",
# # # # # #     )
# # # # # #     args = parser.parse_args()

# # # # # #     df = generate(
# # # # # #         n_rows=args.rows,
# # # # # #         n_plants=args.plants,
# # # # # #         n_months=args.months,
# # # # # #         n_machines=args.machines,
# # # # # #         seed=args.seed,
# # # # # #     )
# # # # # #     df.to_csv(args.out, index=False)

# # # # # #     print(f"Wrote {args.out}")
# # # # # #     print(
# # # # # #         f"Rows: {len(df)} | "
# # # # # #         f"Plants: {df.plant_id.nunique()} | "
# # # # # #         f"Categories: {df.work_order_type.nunique()} | "
# # # # # #         f"Months: {df.notification_date.str[:7].nunique()}"
# # # # # #     )
# # # # # #     print("Columns:", list(df.columns))


# # # # # # if __name__ == "__main__":
# # # # # #     main()










# from dotenv import load_dotenv
# import os
# import requests

# load_dotenv()

# host = os.environ["DATABRICKS_HOST"].rstrip("/")
# token = os.environ["DATABRICKS_TOKEN"]

# response = requests.get(
#     f"{host}/api/2.0/preview/scim/v2/Me",
#     headers={"Authorization": f"Bearer {token}"},
# )

# print(response.status_code)
# print(response.json())




"""
Databricks Unity Catalog permission checker.

Run this locally (same machine/env as your FastAPI app) to find out exactly
why the Files API is returning 403 Forbidden.

Usage:
    set DATABRICKS_HOST=https://adb-xxxx.azuredatabricks.net      (Windows)
    set DATABRICKS_TOKEN=dapiXXXXXXXXXXXXXXXX
    python check_databricks_permissions.py --catalog veriton-db --schema landing --volume datasets

    # optionally also test a specific file path directly:
    python check_databricks_permissions.py --catalog veriton-db --schema landing --volume datasets \
        --file "/Volumes/veriton-db/landing/datasets/34bc6e3a-2076-49ce-a15e-fe5a4ad6e68d/1f4b0c2eba954bf79545aa64965b02f4/Sample-dataset.csv"

It does NOT need DATABRICKS_CLUSTER_ID - none of these checks use compute,
they all hit REST APIs directly (same as the app's databricks_handler.py).
"""

import os
import sys
import json
import argparse
import requests
from dotenv import load_dotenv
load_dotenv()

def get_env_or_exit(name: str) -> str:
    val = os.getenv(name)
    if not val:
        print(f"ERROR: environment variable {name} is not set.")
        sys.exit(1)
    return val


def pretty(title: str, resp: requests.Response):
    print(f"\n--- {title} ---")
    print(f"Status: {resp.status_code}")
    try:
        body = resp.json()
        print(json.dumps(body, indent=2))
    except ValueError:
        print(resp.text[:2000])


def check_identity(host: str, headers: dict) -> str:
    """Who does this token actually authenticate as?"""
    url = f"{host}/api/2.0/preview/scim/v2/Me"
    r = requests.get(url, headers=headers, timeout=30)
    pretty("Token identity (SCIM /Me)", r)
    if r.status_code != 200:
        print(
            "\n>>> Token is not even authenticating. It is likely expired, "
            "revoked, or PATs are disabled for this workspace. Regenerate "
            "the token under User Settings -> Developer -> Access Tokens "
            "before checking anything else.\n"
        )
        return ""
    body = r.json()
    identity = body.get("userName") or body.get("emails", [{}])[0].get("value", "")
    print(f"\n>>> This token authenticates as: {identity!r}\n")
    return identity


def check_permissions(host: str, headers: dict, securable_type: str, full_name: str):
    """
    Unity Catalog permissions API - shows exactly which grants exist and
    who they're granted to, for one securable (catalog / schema / volume).
    Docs: GET /api/2.1/unity-catalog/permissions/{securable_type}/{full_name}
    """
    url = f"{host}/api/2.1/unity-catalog/permissions/{securable_type}/{full_name}"
    r = requests.get(url, headers=headers, timeout=30)
    pretty(f"Grants on {securable_type.upper()} '{full_name}'", r)
    if r.status_code == 200:
        assignments = r.json().get("privilege_assignments", [])
        if not assignments:
            print(f">>> No grants at all exist on this {securable_type}. "
                  f"Nobody has explicit access - an admin needs to GRANT.")
        return r.json()
    elif r.status_code == 403:
        print(f">>> Your token cannot even VIEW grants on this {securable_type}. "
              f"You likely need an admin to run this check instead, or to "
              f"grant you access directly.")
    elif r.status_code == 404:
        print(f">>> {securable_type} '{full_name}' does not exist (check spelling/case).")
    return None


def check_file_head(host: str, headers: dict, file_path: str):
    """
    Mirrors databricks_handler.file_exists() / download_file() - a HEAD
    request tells you 200 (readable), 404 (not found), or 403 (exists but
    no permission) without downloading the whole file.
    """
    clean = "/" + file_path.strip().strip("/")
    if not clean.lower().startswith("/volumes/"):
        print(f"ERROR: --file must start with /Volumes/, got: {file_path}")
        return
    clean = "/Volumes/" + clean[len("/volumes/"):]
    url = f"{host}/api/2.0/fs/files{clean}"
    r = requests.head(url, headers=headers, timeout=30)
    pretty(f"HEAD on file {clean}", r)
    if r.status_code == 200:
        print(">>> File exists AND is readable with this token. If the app "
              "still 403s, the app process may be using a different token "
              "than the one in this shell - check for a stale .env being "
              "loaded, or a second DATABRICKS_TOKEN set elsewhere.")
    elif r.status_code == 404:
        print(">>> File does not exist at this exact path (check job-id folder / filename).")
    elif r.status_code == 403:
        print(">>> File exists but this token's identity lacks READ VOLUME "
              "on the containing volume (or USE CATALOG / USE SCHEMA higher up).")


def main():
    parser = argparse.ArgumentParser(description="Diagnose Databricks Unity Catalog 403s")
    parser.add_argument("--catalog", required=True, help="e.g. veriton-db")
    parser.add_argument("--schema", required=True, help="e.g. landing")
    parser.add_argument("--volume", required=True, help="e.g. datasets")
    parser.add_argument("--file", default=None, help="Optional full /Volumes/... file path to test directly")
    args = parser.parse_args()

    host = get_env_or_exit("DATABRICKS_HOST").rstrip("/")
    token = get_env_or_exit("DATABRICKS_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}

    print(f"Host: {host}")

    identity = check_identity(host, headers)

    check_permissions(host, headers, "catalog", args.catalog)
    check_permissions(host, headers, "schema", f"{args.catalog}.{args.schema}")
    check_permissions(host, headers, "volume", f"{args.catalog}.{args.schema}.{args.volume}")

    if args.file:
        check_file_head(host, headers, args.file)

    print("\n===================================================")
    print("SUMMARY")
    print("===================================================")
    print(f"Token identity : {identity or 'UNKNOWN - token failed to authenticate'}")
    print(f"Catalog        : {args.catalog}")
    print(f"Schema         : {args.catalog}.{args.schema}")
    print(f"Volume         : {args.catalog}.{args.schema}.{args.volume}")
    print(
        "\nIf any of the three grant checks above showed no assignment for "
        f"'{identity}' with USE CATALOG / USE SCHEMA / READ VOLUME respectively, "
        "that is the fix: have a Unity Catalog admin (or metastore admin) run:\n"
    )
    print(f"  GRANT USE CATALOG ON CATALOG `{args.catalog}` TO `{identity or '<identity>'}`;")
    print(f"  GRANT USE SCHEMA ON SCHEMA `{args.catalog}`.{args.schema} TO `{identity or '<identity>'}`;")
    print(f"  GRANT READ VOLUME ON VOLUME `{args.catalog}`.{args.schema}.{args.volume} TO `{identity or '<identity>'}`;")


if __name__ == "__main__":
    main()