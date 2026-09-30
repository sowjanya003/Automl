"""
Snowflake handler for fetching files from named internal stages.

Accepts a Snowflake stage file reference in EITHER of two forms - callers
never need to convert between them, both are parsed to the same internal
(database, schema, stage, relative_path) shape before use:

  URI form (recommended for anything passed through Form(...) fields,
  JSON payloads, query strings, or logged - plain characters only, no '@'
  or quotes to worry about getting mangled in transport):

      snowflake://stage/<DATABASE>/<SCHEMA>/<STAGE_NAME>/<path...>/<file>
      e.g. snowflake://stage/VERITON_DB/DATASETS/DATASETS_STAGE/35/35/sample_test_dataset.csv

  Native Snowflake form (Snowflake's own syntax, e.g. as copy-pasted from
  a Snowflake worksheet or the web UI) - both quoted and unquoted
  identifiers are accepted:

      @"VERITON_DB"."DATASETS"."DATASETS_STAGE"/35/35/sample_test_dataset.csv
      @VERITON_DB.DATASETS.DATASETS_STAGE/35/35/sample_test_dataset.csv

Whichever form is given, the actual SQL sent to Snowflake always uses
freshly-built, fully-quoted identifiers (quoting is always safe - it
never changes behavior for already-uppercase names, and correctly
preserves case for any identifier that isn't purely uppercase
alphanumeric+underscore).
"""

import os
import re
import glob
import gzip
import logging
import tempfile
from typing import Optional

import snowflake.connector

logger = logging.getLogger(__name__)

URI_PREFIX = "snowflake://stage/"

# URI form: snowflake://stage/DB/SCHEMA/STAGE/path/to/file.csv
_URI_PATTERN = re.compile(r"^snowflake://stage/([^/]+)/([^/]+)/([^/]+)/(.+)$")

# Root-only form: snowflake://stage/DB/SCHEMA/STAGE (no trailing file path) -
# used for parsing an OUTPUT ROOT reference, as opposed to a full file
# reference. Deliberately separate from _URI_PATTERN, which requires at
# least one path segment after the stage name.
_ROOT_PATTERN = re.compile(r"^snowflake://stage/([^/]+)/([^/]+)/([^/]+)/?$")

# Native form: @"DB"."SCHEMA"."STAGE"/path  or  @DB.SCHEMA.STAGE/path
# (quoted and unquoted identifiers both accepted, may be mixed)
_NATIVE_PATTERN = re.compile(
    r'^@(?:"(?P<db_q>[^"]+)"|(?P<db_u>[A-Za-z0-9_]+))'
    r'\.(?:"(?P<schema_q>[^"]+)"|(?P<schema_u>[A-Za-z0-9_]+))'
    r'\.(?:"(?P<stage_q>[^"]+)"|(?P<stage_u>[A-Za-z0-9_]+))'
    r'/(?P<path>.+)$'
)


class SnowflakeReferenceError(ValueError):
    """Raised when a Snowflake stage reference (either form) is malformed."""
    pass


class SnowflakeFileNotFoundError(FileNotFoundError):
    """Raised when the requested file does not exist on the stage, or the
    stage/schema/database itself is missing or inaccessible."""
    pass


def is_snowflake_reference(file_path: str) -> bool:
    """True for either supported form: snowflake://stage/... or @..."""
    stripped = file_path.strip()
    return stripped.startswith(URI_PREFIX) or stripped.startswith("@")


# Kept as an alias for backward compatibility with existing call sites
# that were written against the URI-only version of this module.
is_snowflake_uri = is_snowflake_reference


def parse_snowflake_reference(file_path: str) -> dict:
    """
    Parse either supported form into
        {"database": ..., "schema": ..., "stage": ..., "relative_path": ...}
    Raises SnowflakeReferenceError on malformed input - never silently
    misparses either form.
    """
    stripped = file_path.strip()

    if stripped.startswith(URI_PREFIX):
        match = _URI_PATTERN.match(stripped)
        if not match:
            raise SnowflakeReferenceError(
                f"Malformed Snowflake URI: '{file_path}'. Expected shape: "
                f"{URI_PREFIX}<DATABASE>/<SCHEMA>/<STAGE>/<path...>/<file>"
            )
        database, schema, stage, relative_path = match.groups()
        return {
            "database": database,
            "schema": schema,
            "stage": stage,
            "relative_path": relative_path,
        }

    if stripped.startswith("@"):
        match = _NATIVE_PATTERN.match(stripped)
        if not match:
            raise SnowflakeReferenceError(
                f"Malformed Snowflake native reference: '{file_path}'. "
                'Expected shape: @"DATABASE"."SCHEMA"."STAGE"/<path...>/<file> '
                "(quotes optional per identifier)"
            )
        groups = match.groupdict()
        return {
            "database": groups["db_q"] or groups["db_u"],
            "schema": groups["schema_q"] or groups["schema_u"],
            "stage": groups["stage_q"] or groups["stage_u"],
            "relative_path": groups["path"],
        }

    raise SnowflakeReferenceError(
        f"Not a recognized Snowflake reference: '{file_path}'. Expected "
        f"either '{URI_PREFIX}...' or a native '@\"DB\".\"SCHEMA\".\"STAGE\"/...' reference."
    )


def parse_snowflake_root(output_root: str) -> dict:
    """
    Parse a bare output-root reference (database/schema/stage only, no
    file path) into {"database": ..., "schema": ..., "stage": ...}.

    Distinct from parse_snowflake_reference(), which requires a trailing
    file path and would reject a root-only reference like
    'snowflake://stage/DB/SCHEMA/STAGE' as malformed.
    """
    stripped = output_root.strip()
    match = _ROOT_PATTERN.match(stripped)
    if not match:
        raise SnowflakeReferenceError(
            f"Malformed Snowflake output root: '{output_root}'. Expected "
            f"shape: {URI_PREFIX}<DATABASE>/<SCHEMA>/<STAGE>"
        )
    database, schema, stage = match.groups()
    return {"database": database, "schema": schema, "stage": stage}


def _quote_identifier(identifier: str) -> str:
    """Always quote - safe default, works for any identifier regardless
    of the case/characters it was originally given in."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def to_native_stage_path(file_path: str) -> str:
    """
    Build Snowflake's native '@"DB"."SCHEMA"."STAGE"/path' syntax from
    EITHER supported input form. Always re-quotes identifiers freshly,
    regardless of whether the input was already in native form and
    already quoted or not.
    """
    parts = parse_snowflake_reference(file_path)
    return (
        f'@{_quote_identifier(parts["database"])}'
        f'.{_quote_identifier(parts["schema"])}'
        f'.{_quote_identifier(parts["stage"])}'
        f'/{parts["relative_path"]}'
    )


# Backward-compatible alias
parse_snowflake_uri = parse_snowflake_reference


class SnowflakeHandler:
    def __init__(
        self,
        account: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        warehouse: Optional[str] = None,
        role: Optional[str] = None,
    ):
        self.account = account or os.getenv("SNOWFLAKE_ACCOUNT")
        self.user = user or os.getenv("SNOWFLAKE_USER")
        self.password = password or os.getenv("SNOWFLAKE_PASSWORD")
        self.warehouse = warehouse or os.getenv("SNOWFLAKE_WAREHOUSE")
        self.role = role or os.getenv("SNOWFLAKE_ROLE")  # optional

    def _connect(self):
        kwargs = {
            "account": self.account,
            "user": self.user,
            "password": self.password,
            "warehouse": self.warehouse,
        }
        if self.role:
            kwargs["role"] = self.role
        return snowflake.connector.connect(**kwargs)

    def download_file(self, file_path: str) -> bytes:
        """
        Download a file from a named internal stage, given EITHER
        supported reference form (URI or native). Returns raw bytes.

        Uses the SQL GET command, which downloads the staged file to a
        local directory - Snowflake's connector has no direct "read
        bytes over SQL" for stage files, so this writes to a temp dir
        and reads it back. The temp dir is auto-deleted as soon as this
        function returns (nothing is left on disk afterward).

        GET auto-compresses downloads by default (adds a .gz suffix)
        unless the file was staged with AUTO_COMPRESS=FALSE, so both
        cases are handled here.
        """
        parts = parse_snowflake_reference(file_path)
        native_path = to_native_stage_path(file_path)
        filename = parts["relative_path"].rstrip("/").split("/")[-1]

        conn = self._connect()
        try:
            cursor = conn.cursor()
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Windows temp paths (C:\Users\...\Temp\tmpXXXX) contain
                    # backslashes and a drive-letter colon. Snowflake treats
                    # backslash as a SQL string-literal escape character, so
                    # a raw Windows path would get mangled (\t -> tab,
                    # \U/\A/\L silently dropped, etc). Convert to forward
                    # slashes and quote the whole location as a string
                    # literal so no character in the path can be
                    # misinterpreted as SQL syntax. Do NOT add an extra
                    # leading slash: a POSIX tmp dir already starts with
                    # "/" (giving the correct file:///tmp/... form), while
                    # a Windows drive-letter path must NOT get one
                    # (file://C:/Users/... is correct; file:///C:/Users/...
                    # is not a valid local path there).
                    local_dir = tmp_dir.replace("\\", "/")
                    if not local_dir.endswith("/"):
                        local_dir += "/"

                    get_command = f"GET {native_path} 'file://{local_dir}'"
                    logger.info(f"Executing Snowflake GET: {get_command}")

                    try:
                        cursor.execute(get_command)
                        results = cursor.fetchall()
                    except snowflake.connector.errors.ProgrammingError as e:
                        # Snowflake raises a ProgrammingError (SQL
                        # compilation error) for a missing/inaccessible
                        # stage file, database, schema, or stage - surface
                        # all of these as not-found rather than a raw
                        # connector exception.
                        raise SnowflakeFileNotFoundError(
                            f"Stage file not found or inaccessible: {file_path} ({e})"
                        )

                    if not results:
                        raise SnowflakeFileNotFoundError(
                            f"File not found on stage: {file_path} (GET returned no rows)"
                        )

                    # Find whatever actually landed on disk - exact name,
                    # or a .gz-compressed version of it (Snowflake's
                    # default behavior unless AUTO_COMPRESS=FALSE was set
                    # when the file was staged).
                    candidates = glob.glob(os.path.join(tmp_dir, "*"))
                    if not candidates:
                        raise SnowflakeFileNotFoundError(
                            f"GET reported success but no file was written to disk: {file_path}"
                        )

                    local_path = candidates[0]
                    with open(local_path, "rb") as f:
                        raw = f.read()

                    if local_path.endswith(".gz") and not filename.endswith(".gz"):
                        raw = gzip.decompress(raw)

                    return raw
            finally:
                cursor.close()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Write-back: mirror an AutoML-generated artifact (predictions, run
    # artifacts) into a Snowflake stage.
    # ------------------------------------------------------------------
    def upload_file(self, file_path: str, content: bytes) -> None:
        """
        Upload bytes to a named internal stage, given EITHER supported
        reference form (URI or native).

        Uses the SQL PUT command, which uploads from a local file - same
        constraint as GET in reverse. Writes `content` to a temp file,
        PUTs it, then the temp file is auto-deleted. AUTO_COMPRESS=FALSE
        is set explicitly so the file lands on the stage with the exact
        same name/bytes given here, rather than Snowflake silently
        gzip-compressing it and appending .gz.
        """
        parts = parse_snowflake_reference(file_path)
        native_path = to_native_stage_path(file_path)
        filename = parts["relative_path"].rstrip("/").split("/")[-1]

        conn = self._connect()
        try:
            cursor = conn.cursor()
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_path = os.path.join(tmp_dir, filename)
                    with open(local_path, "wb") as f:
                        f.write(content)

                    # Same Windows-path-safety reasoning as download_file():
                    # forward slashes only, whole location quoted as a
                    # string literal so no path character can be
                    # misinterpreted as SQL syntax.
                    uri_path = local_path.replace("\\", "/")

                    put_command = (
                        f"PUT 'file://{uri_path}' {native_path.rsplit('/', 1)[0]}/ "
                        f"AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
                    )
                    logger.info(f"Executing Snowflake PUT: {put_command}")

                    try:
                        cursor.execute(put_command)
                        results = cursor.fetchall()
                        # Look up the "status" column by name rather than a
                        # hardcoded index - PUT's result columns are
                        # (source, target, source_size, target_size,
                        # source_compression, target_compression, status,
                        # message), so "status" is index 6, NOT index 1
                        # (index 1 is "target", the uploaded filename -
                        # relying on a hardcoded index previously caused
                        # this check to compare against the filename
                        # instead of the actual status).
                        column_names = [desc[0].lower() for desc in cursor.description]
                        status_idx = column_names.index("status") if "status" in column_names else None
                    except snowflake.connector.errors.ProgrammingError as e:
                        raise SnowflakeFileNotFoundError(
                            f"Stage write-back failed (schema/stage may not exist "
                            f"or is inaccessible): {file_path} ({e})"
                        )

                    if not results:
                        raise RuntimeError(f"PUT returned no result rows for {file_path}")
                    if status_idx is not None and results[0][status_idx] != "UPLOADED":
                        raise RuntimeError(
                            f"PUT did not report UPLOADED status for {file_path}: {results[0][status_idx]}"
                        )
            finally:
                cursor.close()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema / Stage auto-provisioning
    #
    # Assumes the DATABASE already exists (creating a database is a
    # governance decision left to a Snowflake admin, done once manually -
    # same reasoning as leaving CATALOG creation to a Databricks admin).
    # Schema and Stage are lower-stakes, scoped inside an already-approved
    # database, so the app can create them on first use if missing.
    # ------------------------------------------------------------------
    def schema_exists(self, database: str, schema: str) -> bool:
        conn = self._connect()
        try:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    f"SHOW SCHEMAS LIKE '{schema}' IN DATABASE {_quote_identifier(database)}"
                )
                return len(cursor.fetchall()) > 0
            finally:
                cursor.close()
        finally:
            conn.close()

    def create_schema(self, database: str, schema: str) -> None:
        conn = self._connect()
        try:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    f"CREATE SCHEMA IF NOT EXISTS "
                    f"{_quote_identifier(database)}.{_quote_identifier(schema)}"
                )
            finally:
                cursor.close()
        finally:
            conn.close()

    def stage_exists(self, database: str, schema: str, stage: str) -> bool:
        conn = self._connect()
        try:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    f"SHOW STAGES LIKE '{stage}' IN SCHEMA "
                    f"{_quote_identifier(database)}.{_quote_identifier(schema)}"
                )
                return len(cursor.fetchall()) > 0
            finally:
                cursor.close()
        finally:
            conn.close()

    def create_stage(self, database: str, schema: str, stage: str) -> None:
        conn = self._connect()
        try:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    f"CREATE STAGE IF NOT EXISTS "
                    f"{_quote_identifier(database)}.{_quote_identifier(schema)}.{_quote_identifier(stage)}"
                )
            finally:
                cursor.close()
        finally:
            conn.close()

    def ensure_output_stage_exists(self, output_root: str) -> None:
        """
        Given an output root reference (URI or native form), create the
        schema and/or stage if they don't already exist. Assumes
        <database> already exists. Raises if creation fails for a reason
        other than already-exists - callers should catch and treat as
        non-fatal, same as any other write-back failure.
        """
        parts = parse_snowflake_root(output_root)
        database, schema, stage = parts["database"], parts["schema"], parts["stage"]

        if not self.schema_exists(database, schema):
            logger.info(f"Creating missing Snowflake schema: {database}.{schema}")
            self.create_schema(database, schema)

        if not self.stage_exists(database, schema, stage):
            logger.info(f"Creating missing Snowflake stage: {database}.{schema}.{stage}")
            self.create_stage(database, schema, stage)


_snowflake_handler_singleton: Optional[SnowflakeHandler] = None


def get_snowflake_handler() -> Optional[SnowflakeHandler]:
    """
    Lazily builds a module-level SnowflakeHandler singleton from env vars.
    Returns None (rather than raising) if credentials aren't configured,
    so callers can skip Snowflake cleanly instead of crashing at import time.
    """
    global _snowflake_handler_singleton
    if _snowflake_handler_singleton is not None:
        return _snowflake_handler_singleton

    account = os.getenv("SNOWFLAKE_ACCOUNT")
    user = os.getenv("SNOWFLAKE_USER")
    password = os.getenv("SNOWFLAKE_PASSWORD")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")

    if not all([account, user, password, warehouse]):
        logger.warning(
            "Snowflake credentials not configured (SNOWFLAKE_ACCOUNT / "
            "SNOWFLAKE_USER / SNOWFLAKE_PASSWORD / SNOWFLAKE_WAREHOUSE missing)"
        )
        return None

    _snowflake_handler_singleton = SnowflakeHandler(
        account=account, user=user, password=password, warehouse=warehouse
    )
    return _snowflake_handler_singleton
