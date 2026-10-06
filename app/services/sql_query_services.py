import os
import re
import json
import time
import logging
import threading
import warnings
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
from sqlalchemy import exc as sa_exc
from typing import Optional
from pydantic import BaseModel, Field

from app.database import SessionLocal as _MetaSession
from app.services.llm_service import LLMUnavailableError, generate_json, generate_text
from app.services.embedding_service import generate_query_embedding
from app.models import TableMetadataChunk

load_dotenv(override=True)

logger = logging.getLogger(__name__)

SOURCE_DATABASE_URL = os.getenv("SOURCE_DATABASE_URL")
if not SOURCE_DATABASE_URL:
    raise ValueError("SOURCE_DATABASE_URL is not set")

source_engine = create_engine(SOURCE_DATABASE_URL, pool_pre_ping=True)

# Table-metadata embeddings live in the chatbot database (not the source DB),
# so metadata lookups use the shared session factory from app.database
# (imported above as _MetaSession). Its engine sets
# search_path=chatbot,public, which pgvector's "vector" type needs.

# Reflection hits PostgreSQL types SQLAlchemy has no mapping for (point,
# bpchar, ...). They are harmless here (we only need column names and rough
# types), so hide the warning instead of flooding the logs on every request.
warnings.filterwarnings(
    "ignore", message="Did not recognize type", category=sa_exc.SAWarning
)

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

DEFAULT_SCHEMA = "lb1adm"

# Every schema listed here gets ALL its tables added to the catalog
# automatically — this is what avoids hand-writing 60-70 table entries.
# Note the catalog IS the safety allow-list (see _is_safe_select), so this
# lets generated SQL read every table in these schemas. Put anything
# sensitive in EXCLUDE_TABLES, and keep the DB role SELECT-only.
AUTO_DISCOVER_SCHEMAS = ["lb1adm"]
EXCLUDE_TABLES: set[str] = set()

QUERY_TIMEOUT_MS = 30_000

# ---------------------------------------------------------------------------
# TABLE CATALOG — every key is optional:
#   schema   - source schema (defaults to DEFAULT_SCHEMA)
#   summary  - one-line description for table routing. Auto-built from the
#              DB if missing.
#   detail   - hand-written schema notes (gotchas the DB can't tell us,
#              like "this column is always empty"). Auto-built from
#              columns/types/PK read from the database if missing.
#   extra    - extra prompt text added whenever this table is selected
#              (e.g. a required query pattern).
#
# Only these 4 tables have hand-verified detail — real production facts
# (which columns are actually empty, which FKs are confirmed, required
# dedup patterns) that no amount of schema-reflection can produce on its
# own. EVERY OTHER TABLE in lb1adm needs ZERO entry here — it is picked up
# automatically by AUTO_DISCOVER_SCHEMAS and its schema is read straight
# from the database. Add a hand-written entry only when a table turns out
# to need documented gotchas like these.
# ---------------------------------------------------------------------------

_CTE_PATTERN_TEXT = """
REQUIRED pattern for any question that both (a) needs distinct wagons
from p_roams_wagon_master and (b) needs a JOIN, GROUP BY, or aggregate:
dedupe in its own CTE FIRST, then join/aggregate against that CTE — never
mix DISTINCT ON with GROUP BY in the same SELECT, and never aggregate
directly over the raw table without deduping first. Example shape:

  WITH latest_wagons AS (
    SELECT DISTINCT ON (wagon_no) *
    FROM lb1adm.p_roams_wagon_master
    ORDER BY wagon_no, adb_sequence DESC
  )
  SELECT t.wagon_type, COUNT(*) AS wagon_count, t.cc_payload_tons
  FROM latest_wagons w
  JOIN lb1adm.roams_wagon_type_master t ON w.wagon_type = t.wagon_type
  WHERE w.wagon_status = 'A'
  GROUP BY t.wagon_type, t.cc_payload_tons
  ORDER BY wagon_count DESC
  LIMIT 5;

Use this same CTE-first shape for any other dedupe+join+aggregate
question (e.g. counts/lists by owner, by station, by state, by org type).
"""

TABLE_CATALOG = {
    "fmm_org_m": {
        "summary": (
            "Railway organizations/locations — yards, sick lines, depots, "
            "weighbridges, workshops, divisions, headquarters. Has "
            "state/district, and is the central org_slno hub other "
            "tables link into."
        ),
        "detail": """
lb1adm.fmm_org_m — railway organizations of many kinds (yards, sick lines,
depots, weighbridges, workshops, divisions, headquarters, etc.), one row
per organization. This is the central "location/organization" table that
p_roams_wagon_master, roams_owner_master, and roams_wagon_type_master all
link into via org_slno.
  org_slno (text, primary key), org_code, org_desc (name),
  org_type (2-letter code — one of: CD, CP, CR, CS, CT, DV, EC, FD, HQ, MC,
    MF, ML, NP, OM, RB, RD, RH, RU, SD, SH, SL, UT, WB, WD, WS, YD;
    known meanings: WB=Weigh Bridge, YD=Yard, SL=Sick Line, WS=Workshop,
    HQ=Headquarters, DV=Division, RH=Routine Overhaul Depot, CD=Coaching
    Depot, FD=Freight Depot, MF=Manufacturer, UT=Siding/Utility Unit —
    reason about the rest from context if needed; this column references
    a separate org-type lookup table that is NOT queryable here, so treat
    org_type purely as a code, never join to it),
  prnt_org_slno (parent organization), pvt_rly (constrained by the
    database to only ever be exactly 'PVT' or 'RLY', may be NULL),
  valid_flag ('Y'/'N', may be NULL), manufacturer (bool, mostly NULL —
    only set on organizations that are manufacturers), district_name,
  state_name, gis_flag (bool), depot_rb_2017 (bool), depot_rb_2008 (bool),
  incharge_details (jsonb; keys: inchargeName, inchargeDesignation,
    contactNo, officePostalAddress jsonb with city/state/pincode/district/
    addressLine1/addressLine2 — use ->>'key' to extract a field),
  station_name (a station code that references a separate station-master
    table NOT queryable here — treat as a plain code, never join to it),
  man_power_flag ('Y'/'N', NOT NULL), dispatch_for_flag (bool),
  location_type (free-text/inconsistent — values seen include CD,
    PRIVATE, ST, STATE GOVERNMENT, CENTRAL GOVERNMENT, DEMU SHED, MEMU
    SHED, and occasional stray/dirty values; treat with ILIKE and don't
    assume a clean fixed set), org_slno_category (single-letter code,
    mostly NULL), rs_type_attended (text array — use unnest() to inspect
    contents), maintenance_activities_available (text array — use
    unnest() to inspect contents).
""",
    },
    "roams_owner_master": {
        "summary": (
            "Wagon owner records — owner_id, owner_name, type "
            "(PVT/RLY/DEF). Joined from p_roams_wagon_master.wagon_owner."
        ),
        "detail": """
lb1adm.roams_owner_master — wagon owner records, one row per owner.
  owner_id (text, primary key — this is what p_roams_wagon_master.
    wagon_owner actually references, confirmed against real data; do NOT
    join wagon_owner to numeric_code), owner_name, numeric_code (this
    references a separate owner-code lookup table that is NOT queryable
    here — treat as a plain code, never join to it), address,
  type ('PVT', 'RLY', 'DEF', or occasionally a value with stray
    whitespace like 'NA ' — PVT=Private, RLY=Railway-owned, DEF=Defence;
    compare case-insensitively to be safe), valid_flag ('Y'/'N'),
  org_slno (real FK to fmm_org_m.org_slno, confirmed against real data).
""",
    },
    "roams_wagon_type_master": {
        "summary": (
            "Wagon type specifications — dimensions, capacity, codal "
            "life, etc, one row per wagon_type. Joined from "
            "p_roams_wagon_master.wagon_type."
        ),
        "detail": """
lb1adm.roams_wagon_type_master — wagon type specifications, one row per
wagon type.
  wagon_type (text, primary key — this is what p_roams_wagon_master.
    wagon_type references, confirmed against real data), wagon_description,
  wagon_class_group (single-letter code, values include F, O, C, T, H, V,
    N, W, X — exact meaning per letter is not documented, reason from
    wagon_description/rake_type/stock_type if the question needs it),
  tare_tons (numeric), cc_payload_tons (numeric, carrying capacity),
  axle_load (numeric), bearing_type, coupling_type, bogie_type,
  brake_type, first_poh_duration, next_poh_duration, first_roh_duration,
  next_roh_duration, codal_life (integer, in years — the wagon type's
    total codal life span; may be NULL, so guard against that, e.g. with
    WHERE codal_life IS NOT NULL. See the note on
    p_roams_wagon_master.manufacturer_date for how to compute a wagon's
    codal-life-due date — commission_date should NOT be used for this),
  gauge ('BG' or 'NG', a few NULL), avg_no_of_wagon_in_rake,
  wagon_type_display_name, no_of_wheels, drawing_no, valid_flag (bool),
  mcdo_flag, old_wagon_type, being_manufactured (bool), rly_pvt_owned
    (single-letter code: R, P, D, or M — likely Railway/Private/Defence/
    Manufacturer-owned respectively, but treat as a categorical code, may
    be NULL), rake_type (NOT NULL) and stock_type (both very
    high-cardinality free-text-ish codes describing the physical wagon
    family/stock — hundreds of distinct near-unique values, often messy,
    e.g. 'BWTB / Defence'; always use ILIKE for these, never exact match),
  prd_flag ('Y'/'N', some NULL), commercial_notif_date (date), remarks,
  commercial_letter_image_url (jsonb array), speed_consent (jsonb),
  ir_empty_speed, ir_loaded_speed, dfcc_empty_speed, dfcc_loaded_speed,
  org_slno (real FK to fmm_org_m.org_slno, confirmed against real data,
    though only populated on some rows).
""",
    },
    "p_roams_wagon_master": {
        "summary": (
            "Individual physical wagon records, one row per wagon "
            "(deduped by wagon_no — see detail for why). Links to type, "
            "owner, and organization/station."
        ),
        "extra": _CTE_PATTERN_TEXT,
        "detail": """
lb1adm.p_roams_wagon_master — individual wagon records, one row per
physical wagon (but see adb_sequence note below — the table holds
multiple historical versions per wagon, not one row per wagon).
  wagon_no (text — the wagon's actual identifying number, effectively
    unique per physical wagon once deduped; prefer this over wagon_id for
    display/grouping), record_created_person, wagon_type (FK, confirmed,
    to roams_wagon_type_master.wagon_type), po_number, tare (numeric —
    EMPTY for every row in the actual data, verified: 0 of ~21,000 rows
    populated; never expect a real value from this column, any
    aggregate over it will correctly come back NULL),
  manufacturer_date (date — fully populated in the real data, unlike
    commission_date. This is the correct column to use as "when the wagon
    entered service" for any age/codal-life calculation, since
    commission_date is present in the schema but is EMPTY for every row
    in the actual data (verified: 0 of ~23,700 rows have it populated) —
    NEVER use commission_date in a calculation, always use
    manufacturer_date instead. E.g. to compute a wagon's codal-life-due
    date: manufacturer_date + (codal_life || ' years')::interval, joining
    to roams_wagon_type_master for codal_life), original_cost (numeric),
  mf_id, warranty_end_date (date, only ~1/3 of rows populated),
  manufacturer_country, diagram_number, manufacturer_serial_number,
  commission_date (present in the schema but EMPTY for every row in the
    actual data — do not use it for anything; see manufacturer_date
    above), commission_station (also empty/unreliable in practice — do
    not rely on it), wagon_owner (FK, confirmed, to
    roams_owner_master.owner_id — NOT numeric_code),
  wagon_carrying_capacity (numeric), scheme_name,
  pipe_fitted (single character, values 'T' or 'S', some NULL — not a
    plain boolean despite the name), wagon_status (single character code:
    'A', 'R', 'C', or 'I' — 'A' is by far the most common and is almost
    certainly "active/in service"; the exact meaning of R/C/I is not
    documented, so state the raw code if asked rather than guessing its
    meaning, but 'A' is a reasonable default filter for "in service" /
    "currently running" style questions), base_station, roh_stn, poh_stn
    (all three are, in this data, actually FKs to fmm_org_m.org_slno,
    confirmed against real data — NOT to org_code, despite the "station"
    naming), organization_type ('RLY', 'PVT', or 'DEF', matching
    roams_owner_master.type's categories), valid_flag ('Y'/'N', in
    practice always 'Y' in current data), roh_date (date, last ROH date),
  poh_date (date, last POH date), brake_type, rfid1 (bool), rfid2 (bool),
  poh_return_date (date), pvt_vehicle_no, rb_wagon_id, intrm_remarks,
  org_slno (FK, confirmed, to fmm_org_m.org_slno), pvt_owner_code,
  wagon_usage (mostly NULL, occasionally 'DP'), lessor_flag (NULL in all
    current data), agreement_wagon_pool, wagon_no_old,
  update_in_no_type_owner (bool), wagon_id (bigint — NOT reliably unique
    per physical wagon; do not use it as the dedup key), adb_sequence
    (bigint, the column with a real unique index in this table — this
    table stores multiple historical versions of the same wagon_no, and
    adb_sequence is what actually makes each row unique. When counting or
    listing distinct wagons, ALWAYS dedupe first, using the CTE pattern
    shown below, unless the question is explicitly about historical
    versions/changes over time), adb_timestamp (timestamp — when this
    version of the row was recorded; an alternative to adb_sequence for
    picking the latest version).
""",
    },
}

# Hand-verified join keys: (table_a, column_a, table_b, column_b). Foreign
# keys declared in the database are discovered automatically (see
# _get_join_edges), so with 60-70 tables this list only needs entries for
# joins the database itself doesn't declare (no FK constraint, or FK
# pointing to a lookup table that isn't in the catalog).
JOIN_EDGES = [
    ("p_roams_wagon_master", "wagon_type", "roams_wagon_type_master", "wagon_type"),
    ("p_roams_wagon_master", "wagon_owner", "roams_owner_master", "owner_id"),
    ("p_roams_wagon_master", "org_slno", "fmm_org_m", "org_slno"),
    ("p_roams_wagon_master", "base_station", "fmm_org_m", "org_slno"),
    ("p_roams_wagon_master", "roh_stn", "fmm_org_m", "org_slno"),
    ("p_roams_wagon_master", "poh_stn", "fmm_org_m", "org_slno"),
    ("roams_owner_master", "org_slno", "fmm_org_m", "org_slno"),
    ("roams_wagon_type_master", "org_slno", "fmm_org_m", "org_slno"),
]

# If the catalog has this many tables or fewer, no routing is needed and all
# tables go straight into the prompt. With 60-70 tables this is always
# exceeded, so routing is used.
_ROUTING_THRESHOLD = 6

# The metadata vector search picks this many tables closest to the question
# (join targets are added on top of these afterwards).
METADATA_TOP_K = 3

# A text column is treated as "categorical" (and its distinct values are
# shown to the model) only if it has this many distinct values or fewer.
# Columns like wagon_no or remarks have thousands of distinct values and are
# skipped; only small enumerations (type, status, gauge, org_type, ...) qualify.
LOW_CARDINALITY_MAX_DISTINCT = 50

MAX_RESULT_ROWS = 500

_FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "truncate", "grant",
    "revoke", "create", "exec", "execute", "call", "copy", "merge", "into",
]
_FORBIDDEN_WORD_RE = re.compile(r"\b(?:" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b")
_FORBIDDEN_TOKENS = ("--", "/*")

# ---------------------------------------------------------------------------
# CATALOG HELPERS — discovery, schema lookup, reflection from the database.
# This is what replaces hand-writing 60-70 table entries.
# ---------------------------------------------------------------------------

_catalog_lock = threading.Lock()
_catalog_ready = False

_reflect_lock = threading.Lock()
_reflect_cache: dict[str, dict] = {}

_fk_lock = threading.Lock()
_fk_edges_cache: Optional[list] = None

# Each table's low-cardinality values are fetched once and cached here, so
# the database is not queried again for every question.
_value_lock = threading.Lock()
_value_cache: dict[str, dict[str, list[str]]] = {}


def _ensure_catalog_ready() -> None:
    """
    Adds every table from AUTO_DISCOVER_SCHEMAS to the catalog (once).
    This is the actual "don't hardcode 60-70 tables" mechanism: whatever
    exists in lb1adm today, or gets added tomorrow, appears here with no
    code change — sync_master_data.py's own discovery is independent of
    this, this only controls what generated SQL is allowed to query.
    """
    global _catalog_ready
    if _catalog_ready:
        return
    with _catalog_lock:
        if _catalog_ready:
            return
        try:
            if AUTO_DISCOVER_SCHEMAS:
                inspector = inspect(source_engine)
                for schema in AUTO_DISCOVER_SCHEMAS:
                    for table in inspector.get_table_names(schema=schema):
                        if table not in TABLE_CATALOG and table not in EXCLUDE_TABLES:
                            TABLE_CATALOG[table] = {"schema": schema}
            _catalog_ready = True
        except Exception as e:
            logger.warning(f"Table auto-discovery failed: {e}")


def _schema_of(table: str) -> str:
    return TABLE_CATALOG[table].get("schema", DEFAULT_SCHEMA)


def _get_reflected(table: str) -> dict:
    """Columns / primary key / comment for a table, read from the DB once and cached."""
    with _reflect_lock:
        cached = _reflect_cache.get(table)
    if cached is not None:
        return cached

    meta = {"columns": [], "pk": [], "comment": None}
    schema = _schema_of(table)
    try:
        inspector = inspect(source_engine)
        for col in inspector.get_columns(table, schema=schema):
            try:
                type_name = str(col["type"])
            except Exception:
                type_name = "unknown"
            meta["columns"].append((col["name"], type_name))
        meta["pk"] = inspector.get_pk_constraint(table, schema=schema).get("constrained_columns") or []
        try:
            meta["comment"] = inspector.get_table_comment(table, schema=schema).get("text")
        except Exception:
            meta["comment"] = None
    except Exception as e:
        logger.warning(f"Could not read schema of {schema}.{table}: {e}")

    with _reflect_lock:
        _reflect_cache[table] = meta
    return meta


# Text/varchar SQLAlchemy type names worth a low-cardinality check. Numeric,
# date, boolean, jsonb and array types are skipped: they never have a
# "Northern Railway" vs "NR" style mismatch.
_CATEGORICAL_TYPE_PREFIXES = ("VARCHAR", "CHAR", "TEXT", "ENUM")


def _get_low_cardinality_values(table: str) -> dict[str, list[str]]:
    """
    For every text column of the table, returns its distinct values if
    there are LOW_CARDINALITY_MAX_DISTINCT or fewer. This lets the model map
    "Northern Railway" in a question to the value actually stored in the DB
    (e.g. "NR") without embedding or chunking any row data. It is a cheap
    live lookup, only on small enumeration-like columns, run once per table
    and then cached.

    The query is bounded: GROUP BY ... LIMIT (max+1) means a
    high-cardinality column (wagon_no, remarks, address) never needs a full
    distinct count; as soon as more than the limit is seen, the column is
    skipped.
    """
    with _value_lock:
        cached = _value_cache.get(table)
    if cached is not None:
        return cached

    meta = _get_reflected(table)
    schema = _schema_of(table)
    preparer = source_engine.dialect.identifier_preparer
    table_sql = f"{preparer.quote_schema(schema)}.{preparer.quote(table)}"

    result: dict[str, list[str]] = {}
    try:
        with source_engine.connect() as conn:
            for col_name, type_name in meta["columns"]:
                if not type_name.upper().startswith(_CATEGORICAL_TYPE_PREFIXES):
                    continue
                col_sql = preparer.quote(col_name)
                try:
                    rows = conn.execute(
                        text(
                            f"SELECT {col_sql} FROM {table_sql} "
                            f"WHERE {col_sql} IS NOT NULL "
                            f"GROUP BY {col_sql} "
                            f"LIMIT {LOW_CARDINALITY_MAX_DISTINCT + 1}"
                        )
                    ).fetchall()
                except Exception as e:
                    logger.warning(f"Distinct-value check failed for {table}.{col_name}: {e}")
                    continue

                if 0 < len(rows) <= LOW_CARDINALITY_MAX_DISTINCT:
                    result[col_name] = sorted(str(r[0]) for r in rows)
    except Exception as e:
        logger.warning(f"Low-cardinality scan failed for {table}: {e}")

    with _value_lock:
        _value_cache[table] = result
    return result


def _get_table_summary(table: str) -> str:
    """One-line description used for cheap table routing (see _ROUTING_THRESHOLD)."""
    info = TABLE_CATALOG[table]
    if info.get("summary"):
        return info["summary"]

    meta = _get_reflected(table)
    parts = []
    if meta["comment"]:
        parts.append(meta["comment"].strip())
    if meta["columns"]:
        parts.append("Columns: " + ", ".join(name for name, _ in meta["columns"][:15]))
    return " ".join(parts) or f"Table {table}"


def _get_table_detail(table: str) -> str:
    """Full schema text: hand-written if provided, otherwise built entirely from the DB."""
    info = TABLE_CATALOG[table]
    if info.get("detail"):
        base = info["detail"]
    else:
        meta = _get_reflected(table)
        description = info.get("summary") or meta["comment"] or "no description available"
        lines = [f"{_schema_of(table)}.{table} — {description}"]
        if meta["pk"]:
            lines.append(f"  primary key: {', '.join(meta['pk'])}")
        if meta["columns"]:
            lines.append("  columns: " + ", ".join(f"{name} ({type_name})" for name, type_name in meta["columns"]))
        if info.get("notes"):
            lines.append("  notes: " + info["notes"].strip())
        base = "\n".join(lines) + "\n"

    # Append the actually stored values of small enumeration-like text columns
    # (type, status, gauge, org_type, ...). This makes the model write
    # `= 'NR'` instead of `= 'Northern Railway'` when the question uses the
    # full name but the DB stores a short code.
    categorical_values = _get_low_cardinality_values(table)
    if categorical_values:
        value_lines = [
            f"  {col} actual stored values: {', '.join(repr(v) for v in values)}"
            for col, values in categorical_values.items()
        ]
        base += "\n" + "\n".join(value_lines) + "\n"

    return base


# ---------------------------------------------------------------------------
# METADATA EMBEDDING — only table metadata (name, description, columns) is
# embedded, never row data. sync_master_data.py takes the text from here,
# embeds it and stores it in chatbot.table_metadata_chunks.
# ---------------------------------------------------------------------------

def _build_metadata_text(table: str) -> str:
    """Metadata text for one table: schema-qualified name, description, column names."""
    info = TABLE_CATALOG[table]
    meta = _get_reflected(table)

    description = info.get("summary") or (meta["comment"] or "").strip()
    column_names = ", ".join(name for name, _ in meta["columns"])

    parts = [f"Table: {_schema_of(table)}.{table}"]
    if description:
        parts.append(f"Description: {description}")
    if column_names:
        parts.append(f"Columns: {column_names}")
    return "\n".join(parts)


def get_catalog_metadata_texts() -> dict[str, str]:
    """
    Returns {"schema.table": metadata_text} for every table in the catalog.
    Called by sync_master_data.py.
    """
    _ensure_catalog_ready()
    return {
        f"{_schema_of(table)}.{table}": _build_metadata_text(table)
        for table in list(TABLE_CATALOG)
    }


def _search_tables_by_metadata(question: str, k: int = METADATA_TOP_K) -> set[str]:
    """
    Finds the k tables closest to the question embedding (cosine distance
    over table_metadata_chunks). On any error, or if the table is empty,
    returns an empty set so the caller can fall back to LLM routing.
    """
    try:
        question_embedding = generate_query_embedding(question)
        db = _MetaSession()
        try:
            rows = (
                db.query(TableMetadataChunk.source_table)
                .filter(TableMetadataChunk.embedding.isnot(None))
                .order_by(TableMetadataChunk.embedding.cosine_distance(question_embedding))
                .limit(k)
                .all()
            )
        finally:
            db.close()

        picked: set[str] = set()
        for (ref,) in rows:
            schema, _, table = ref.partition(".")
            if table in TABLE_CATALOG and _schema_of(table) == schema:
                picked.add(table)
        return picked
    except Exception as e:
        logger.warning(f"Metadata vector search failed, falling back to LLM routing: {e}")
        return set()


def _get_join_edges() -> list:
    """Hand-verified JOIN_EDGES plus single-column foreign keys declared in the DB itself."""
    global _fk_edges_cache
    _ensure_catalog_ready()

    with _fk_lock:
        if _fk_edges_cache is None:
            discovered = []
            try:
                inspector = inspect(source_engine)
                for table in list(TABLE_CATALOG):
                    for fk in inspector.get_foreign_keys(table, schema=_schema_of(table)):
                        if len(fk["constrained_columns"]) != 1 or len(fk["referred_columns"]) != 1:
                            continue
                        referred = fk["referred_table"]
                        if referred in TABLE_CATALOG:
                            discovered.append(
                                (table, fk["constrained_columns"][0], referred, fk["referred_columns"][0])
                            )
            except Exception as e:
                logger.warning(f"Foreign-key discovery failed: {e}")
            _fk_edges_cache = discovered
        discovered = _fk_edges_cache

    merged = []
    seen = set()
    for a, ca, b, cb in list(JOIN_EDGES) + discovered:
        key = frozenset([(a, ca), (b, cb)])
        if key not in seen:
            seen.add(key)
            merged.append((a, ca, b, cb))
    return merged


def _build_join_text(tables: set[str]) -> str:
    lines = [
        f"  {a}.{ca} = {b}.{cb}"
        for a, ca, b, cb in _get_join_edges()
        if a in tables and b in tables
    ]
    if not lines:
        return ""
    return (
        "\nVerified join keys (use these, not other same-named-looking columns):\n"
        + "\n".join(lines)
        + "\nAny question that needs data from more than one of the tables above"
        " should use these joins.\n"
    )


class TableSelection(BaseModel):
    relevant_tables: list[str] = Field(
        description=(
            "Names of tables (must be exactly the names given in the "
            "catalog) that are relevant to answering this question — "
            "include a table if the question needs data from it directly, "
            "or needs it to join to another relevant table. If unsure, "
            "err on the side of including a table rather than excluding "
            "it."
        )
    )


def _select_relevant_tables(question: str) -> set[str]:
    """
    This is what keeps the prompt from ballooning with 60-70 tables: only
    the full schema of the 2-3 relevant tables goes into the SQL-writing
    prompt.

    The first routing method is METADATA VECTOR SEARCH (matching the
    question embedding against table_metadata_chunks), which needs no LLM
    call. If it finds nothing (empty table, sync not run, or an error), the
    older LLM routing is used as a fallback, and if that fails too, all
    tables are used.

    After selection, join targets are added so cross-table questions do not
    miss a table they need.
    """
    _ensure_catalog_ready()

    if len(TABLE_CATALOG) <= _ROUTING_THRESHOLD:
        return set(TABLE_CATALOG.keys())

    selected = _search_tables_by_metadata(question)

    if not selected:
        catalog_text = "\n".join(
            f"- {name}: {_get_table_summary(name)}" for name in TABLE_CATALOG
        )

        prompt = f"""
        Here is a catalog of available database tables:

        {catalog_text}

        Question: {question}

        Which of these tables (by exact name) are relevant to answering this
        question, either directly or as a join target for another relevant
        table?
        """

        try:
            selected = set(generate_json(prompt, TableSelection).get("relevant_tables") or [])
            selected = {t for t in selected if t in TABLE_CATALOG}
        except Exception as e:
            logger.warning(f"Table selection failed, falling back to full catalog: {e}")
            selected = set()

    if not selected:
        return set(TABLE_CATALOG.keys())

    expanded = set(selected)
    for a, _ca, b, _cb in _get_join_edges():
        if a in selected:
            expanded.add(b)
        if b in selected:
            expanded.add(a)

    return expanded


def _build_schema_text(tables: set[str]) -> str:
    parts = [_get_table_detail(t) for t in TABLE_CATALOG if t in tables]

    join_text = _build_join_text(tables)
    if join_text:
        parts.append(join_text)

    for t in TABLE_CATALOG:
        if t in tables and TABLE_CATALOG[t].get("extra"):
            parts.append(TABLE_CATALOG[t]["extra"])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# SQL SAFETY
# ---------------------------------------------------------------------------

_SCAN_NOISE_RE = re.compile(r"\b(extract|substring|trim|overlay)\s*\(([^()]*?)\bfrom\b")
_TABLE_REF_RE = re.compile(r"\b(?:from|join)\s+(?:(\w+)\.)?(\w+)")


def _referenced_tables(sql: str) -> list[tuple[Optional[str], str]]:
    lowered = sql.strip().rstrip(";").strip().lower()

    scan_text = _SCAN_NOISE_RE.sub(r"\1(\2", lowered)
    scan_text = re.sub(r"\bdistinct\s+from\b", "distinct", scan_text)

    cte_names = set(re.findall(r"(\w+)\s+as\s*\(", lowered))

    seen = set()
    ordered = []
    for schema, table in _TABLE_REF_RE.findall(scan_text):
        schema = schema or None
        if schema is None and table in cte_names:
            continue
        key = (schema, table)
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _is_safe_select(sql: str) -> bool:
    """
    Validates that `sql` is a single, read-only SELECT touching only
    catalog tables. Checked against the FULL catalog (all 60-70 discovered
    tables), regardless of which subset was shown to the model.
    """
    _ensure_catalog_ready()

    stripped = sql.strip().rstrip(";").strip()
    lowered = stripped.lower()

    if not lowered.startswith("select") and not lowered.startswith("with"):
        return False
    if ";" in stripped:
        return False
    if any(token in lowered for token in _FORBIDDEN_TOKENS):
        return False
    if _FORBIDDEN_WORD_RE.search(lowered):
        return False

    referenced = _referenced_tables(sql)
    if not referenced:
        return False

    for schema, table in referenced:
        if table not in TABLE_CATALOG:
            return False
        if schema is not None and schema != _schema_of(table).lower():
            return False
    return True


def _extract_referenced_tables(sql: str) -> list[str]:
    seen = set()
    ordered_unique = []
    for _schema, table in _referenced_tables(sql):
        if table not in seen:
            seen.add(table)
            ordered_unique.append(table)
    return ordered_unique


class SQLQueryPlan(BaseModel):
    can_answer_with_sql: bool = Field(
        description=(
            "True if this question can be answered by querying the railway "
            "master-data tables directly — counts, lists, filters, "
            "summaries, distributions, aggregates, lookups by name/code, "
            "date-based calculations, or cross-table joins the schema "
            "below can express. False if the question is about an "
            "uploaded document/manual/procedure, or is a general question "
            "unrelated to this data."
        )
    )
    sql: Optional[str] = Field(
        None,
        description=(
            "A single, read-only PostgreSQL SELECT statement (a WITH ... "
            "SELECT CTE counts as one statement; table names "
            "schema-qualified exactly as shown in the schema, i.e. "
            "<schema>.<table>) that answers the question. Required if "
            "can_answer_with_sql is True; otherwise omit."
        ),
    )


_SQL_CACHE_TTL_SECONDS = 6 * 60 * 60
_SQL_CACHE_MAX_ENTRIES = 500

_sql_cache: dict[str, dict] = {}
_sql_cache_order: list[str] = []
_sql_cache_lock = threading.Lock()


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower()).rstrip("?.! ")


def _get_cached_plan(question: str) -> Optional[dict]:
    key = _normalize_question(question)
    with _sql_cache_lock:
        entry = _sql_cache.get(key)
        if entry is None:
            return None
        if entry["expires_at"] < time.time():
            del _sql_cache[key]
            if key in _sql_cache_order:
                _sql_cache_order.remove(key)
            return None
        return entry["plan"]


def _store_cached_plan(question: str, plan: dict) -> None:
    key = _normalize_question(question)
    with _sql_cache_lock:
        if key not in _sql_cache and len(_sql_cache_order) >= _SQL_CACHE_MAX_ENTRIES:
            oldest_key = _sql_cache_order.pop(0)
            _sql_cache.pop(oldest_key, None)
        if key not in _sql_cache:
            _sql_cache_order.append(key)
        _sql_cache[key] = {
            "plan": plan,
            "expires_at": time.time() + _SQL_CACHE_TTL_SECONDS,
        }


def generate_sql_query(question: str) -> Optional[dict]:
    if not question or not question.strip():
        return None

    cached_plan = _get_cached_plan(question)
    if cached_plan is not None:
        return cached_plan

    _ensure_catalog_ready()

    relevant_tables = _select_relevant_tables(question)
    schema_text = _build_schema_text(relevant_tables)

    prompt = f"""
    You have READ-ONLY access to a PostgreSQL database with these tables:

    {schema_text}

    Question: {question}

    Decide whether this question can be answered by querying the tables
    above. If yes, write ONE single PostgreSQL SELECT statement (a WITH/CTE
    prefix is fine and encouraged — see the required dedupe pattern above,
    if shown) that answers it. Use the verified join keys listed above
    whenever the question spans more than one table — do not invent other
    join columns, and never join to a table that isn't one shown above
    (some columns reference external lookup tables that are explicitly
    called out as not queryable — treat those as plain codes). If a
    required query pattern is shown above for a table, follow it exactly.
    Use unnest() for array columns when their contents matter, and
    ->>'key' for JSON/JSONB fields. Always write table names
    schema-qualified, exactly as shown above. Never write more than one
    statement, and never write anything other than a SELECT (with an
    optional WITH prefix).

    For any comparison against a free-text or high-cardinality categorical
    column, always use case-insensitive matching (ILIKE, not =), since the
    underlying data may be stored in mixed, inconsistent, or slightly
    messy form.

    If none of the tables shown above can answer this question — for
    example if it's about an uploaded document/manual/procedure, or is
    unrelated to this data — set can_answer_with_sql to false and omit
    sql.
    """

    try:
        plan = generate_json(prompt, SQLQueryPlan)
    except LLMUnavailableError:
        # Do not turn an AI-service outage into a fake "no answer found";
        # let the API layer report it as a 503.
        raise
    except Exception as e:
        logger.warning(f"SQL plan generation failed: {e}")
        return None

    if plan.get("can_answer_with_sql") and plan.get("sql"):
        _store_cached_plan(question, plan)

    return plan


def execute_sql_query(sql: str) -> list[dict]:
    cleaned = sql.strip().rstrip(";")
    with source_engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        conn.execute(text(f"SET LOCAL statement_timeout = {int(QUERY_TIMEOUT_MS)}"))
        result = conn.execute(text(cleaned))
        columns = result.keys()
        rows = result.fetchmany(MAX_RESULT_ROWS)
        return [dict(zip(columns, row)) for row in rows]


def summarize_sql_results(question: str, sql: str, rows: list[dict]) -> str:
    if not rows:
        return "NOT_FOUND"

    rows_text = json.dumps(rows, default=str)

    prompt = f"""
    This SQL query was run against the railway master-data database to
    answer the user's question:

    {sql}

    It returned {len(rows)} row(s):
    {rows_text}

    Using ONLY this result, answer the question.

    FORMAT RULE: if the result has more than one row AND more than one
    column, format the answer as a Markdown table (with a header row),
    optionally preceded by one short sentence of context. If the result
    is a single value (e.g. a count) or a single row, answer in a plain
    sentence instead — do not build a table for that case.

    IMPORTANT: rows were actually returned, so this question DOES have an
    answer — do not respond with "NOT_FOUND" just because some requested
    field is NULL/missing for some or all rows. In that case, still answer
    using whatever fields ARE populated, and explicitly say which
    requested field was not available. Reserve "NOT_FOUND" strictly for
    cases where these specific rows are about a completely different
    entity than what was asked — never for missing/NULL values within an
    otherwise relevant result.

    Question: {question}
    """

    try:
        return generate_text(prompt)
    except LLMUnavailableError:
        raise
    except Exception as e:
        logger.warning(f"summarize_sql_results failed: {e}")
        return "NOT_FOUND"


def answer_master_data_query(question: str) -> Optional[dict]:
    plan = generate_sql_query(question)
    if not plan or not plan.get("can_answer_with_sql") or not plan.get("sql"):
        return None

    sql = plan["sql"]
    if not _is_safe_select(sql):
        logger.warning(f"Rejected unsafe or out-of-scope generated SQL: {sql!r}")
        return None

    try:
        rows = execute_sql_query(sql)
    except Exception as e:
        logger.warning(f"SQL execution failed for {sql!r}: {e}")
        return None

    answer = summarize_sql_results(question, sql, rows)
    if answer.strip().upper() == "NOT_FOUND":
        return None

    source_tables = _extract_referenced_tables(sql)
    return {
        "answer": answer,
        "sources": [{
            "type": "master_data",
            "source_table": ", ".join(source_tables) if source_tables else None,
        }],
        "rounds_used": 0,
    }