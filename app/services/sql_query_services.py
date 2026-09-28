import os
import re
import json
import time
import logging
import threading
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from typing import Optional
from pydantic import BaseModel, Field

from app.services.llm_service import client, GEMINI_MODEL

load_dotenv(override=True)

logger = logging.getLogger(__name__)

SOURCE_DATABASE_URL = os.getenv("SOURCE_DATABASE_URL")
if not SOURCE_DATABASE_URL:
    raise ValueError("SOURCE_DATABASE_URL is not set")

# Read-only usage only (see _is_safe_select) — for real safety this should
# also connect as a database role that only has SELECT privilege on these
# tables, so a mistake here can't be the only line of defense.
source_engine = create_engine(SOURCE_DATABASE_URL)

# ---------------------------------------------------------------------------
# TABLE CATALOG — one entry per table, split into a short "summary" (used
# for cheap table-routing when the catalog grows) and a full "detail"
# (the verified schema notes, injected only for tables actually selected as
# relevant to a given question). This is what lets the system scale to many
# more tables/databases later without every prompt growing with the whole
# catalog: only the tables a question actually needs get their full schema
# sent to the model. Nothing about which tables are relevant is hardcoded —
# an LLM call decides that per-question, same as SQL generation itself.
# ---------------------------------------------------------------------------

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

# Verified join keys between tables — used both to build the schema text
# for a question, and to auto-expand a question's selected tables to
# include anything they must join to (see _select_relevant_tables). Add
# new (table_a, table_b) edges here as new tables/joins are verified;
# nothing else needs to change to support them.
JOIN_EDGES = [
    ("p_roams_wagon_master", "roams_wagon_type_master"),
    ("p_roams_wagon_master", "roams_owner_master"),
    ("p_roams_wagon_master", "fmm_org_m"),
    ("roams_owner_master", "fmm_org_m"),
    ("roams_wagon_type_master", "fmm_org_m"),
]

_JOIN_KEY_TEXT = """
Verified join keys (use these, not other same-named-looking columns):
  p_roams_wagon_master.wagon_type   = roams_wagon_type_master.wagon_type
  p_roams_wagon_master.wagon_owner  = roams_owner_master.owner_id
  p_roams_wagon_master.org_slno     = fmm_org_m.org_slno
  p_roams_wagon_master.base_station = fmm_org_m.org_slno
  p_roams_wagon_master.roh_stn      = fmm_org_m.org_slno
  p_roams_wagon_master.poh_stn      = fmm_org_m.org_slno
  roams_owner_master.org_slno       = fmm_org_m.org_slno
  roams_wagon_type_master.org_slno  = fmm_org_m.org_slno
Any question that spans wagons + their type, wagons + their owner, or
wagons/owners/types + the organizations/stations they're linked to should
use these joins.
"""

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

# Only these tables may ever be queried — this is the safety allow-list
# and stays independent of whichever subset gets shown to the model for
# any single question (see _select_relevant_tables below).
_ALLOWED_TABLES = set(TABLE_CATALOG.keys())

# Below this many tables in the catalog, every table's full schema is just
# included directly — the extra routing call would add latency/cost for
# no benefit when there's nothing to filter down. Once the catalog grows
# past this (more tables, more departments' databases), routing kicks in
# automatically with no code change needed here.
_ROUTING_THRESHOLD = 6

# A hard cap on rows fetched, applied regardless of what the generated SQL
# itself asks for — protects against an unexpectedly huge result set
# blowing up the prompt or the response.
MAX_RESULT_ROWS = 500

_FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "truncate", "grant",
    "revoke", "create", "exec", "execute", "call", "copy", "merge",
    "--", "/*", "into",
]


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
    Cheap first pass over just table names + one-line summaries (not full
    schemas) to figure out which tables a question actually needs. This
    is what keeps prompt size bounded as more tables/databases get added
    later — instead of every question paying for every table's full
    schema, only the relevant ones do.

    Below _ROUTING_THRESHOLD tables, this is skipped entirely and all
    tables are used (identical to today's behavior) — the extra LLM round
    trip only starts happening once there's an actual catalog to filter.

    Falls back to "all tables" on any failure or empty result, since
    over-including schema is always safe (just less efficient), while
    under-including could silently break a valid cross-table question.
    """
    if len(TABLE_CATALOG) <= _ROUTING_THRESHOLD:
        return set(TABLE_CATALOG.keys())

    catalog_text = "\n".join(
        f"- {name}: {info['summary']}" for name, info in TABLE_CATALOG.items()
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
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": TableSelection,
            },
        )
        selected = set(json.loads(response.text).get("relevant_tables") or [])
        selected = {t for t in selected if t in TABLE_CATALOG}
    except Exception as e:
        logger.warning(f"Table selection failed, falling back to full catalog: {e}")
        selected = set()

    if not selected:
        return set(TABLE_CATALOG.keys())

    # Expand one hop along verified joins, so a cross-table question isn't
    # broken just because the classifier only named one side of the join.
    expanded = set(selected)
    for a, b in JOIN_EDGES:
        if a in selected:
            expanded.add(b)
        if b in selected:
            expanded.add(a)

    return expanded


def _build_schema_text(tables: set[str]) -> str:
    parts = [TABLE_CATALOG[t]["detail"] for t in TABLE_CATALOG if t in tables]
    if len(tables) > 1:
        parts.append(_JOIN_KEY_TEXT)
    if "p_roams_wagon_master" in tables:
        parts.append(_CTE_PATTERN_TEXT)
    return "\n".join(parts)


def _is_safe_select(sql: str) -> bool:
    """
    Validates that `sql` is a single, read-only SELECT (optionally with a
    leading WITH/CTE) statement touching only known tables. This is the
    actual safety boundary for running LLM-generated SQL — nothing here
    trusts the model's own judgment about what's safe. Checked against the
    FULL catalog's allow-list regardless of which subset was shown to the
    model for this particular question.
    """
    stripped = sql.strip().rstrip(";").strip()
    lowered = stripped.lower()

    if not lowered.startswith("select") and not lowered.startswith("with"):
        return False
    if ";" in stripped:
        # A second statement snuck in after the trailing semicolon we
        # already stripped — reject rather than guess which part is real.
        return False
    if any(kw in lowered for kw in _FORBIDDEN_KEYWORDS):
        return False

    referenced_tables = re.findall(r"(?:from|join)\s+(?:lb1adm\.)?(\w+)", lowered)
    if not referenced_tables:
        return False
    # A CTE name referenced later in the query (e.g. "FROM latest_wagons")
    # is not a real table and must not be checked against _ALLOWED_TABLES.
    cte_names = set(re.findall(r"(\w+)\s+as\s*\(", lowered))
    real_tables = [t for t in referenced_tables if t not in cte_names]
    if not real_tables:
        return False
    return all(t in _ALLOWED_TABLES for t in real_tables)




def _extract_referenced_tables(sql: str) -> list[str]:
    """
    Pulls out the real table names a query touches (excluding CTE names),
    for display purposes — e.g. building a human-readable "source" label.
    Shares the same extraction logic _is_safe_select uses for validation.
    """
    lowered = sql.strip().rstrip(";").strip().lower()
    referenced_tables = re.findall(r"(?:from|join)\s+(?:lb1adm\.)?(\w+)", lowered)
    cte_names = set(re.findall(r"(\w+)\s+as\s*\(", lowered))
    real_tables = [t for t in referenced_tables if t not in cte_names]
    # Preserve first-seen order, drop duplicates
    seen = set()
    ordered_unique = []
    for t in real_tables:
        if t not in seen:
            seen.add(t)
            ordered_unique.append(t)
    return ordered_unique


class SQLQueryPlan(BaseModel):
    can_answer_with_sql: bool = Field(
        description=(
            "True if this question can be answered by querying the railway "
            "master-data tables directly — counts, lists, filters, "
            "summaries, distributions, aggregates, lookups by name/code, "
            "date-based calculations (e.g. codal life due dates), or "
            "cross-table joins the schema below can express, including "
            "questions that require combining information from more than "
            "one of these tables. False if the question is about an "
            "uploaded document/manual/procedure, or is a general question "
            "unrelated to this data."
        )
    )
    sql: Optional[str] = Field(
        None,
        description=(
            "A single, read-only PostgreSQL SELECT statement (a WITH ... "
            "SELECT CTE counts as one statement; table names "
            "schema-qualified as lb1adm.<table>) that answers the "
            "question. Required if can_answer_with_sql is True; otherwise "
            "omit."
        ),
    )


# ---------------------------------------------------------------------------
# SQL generation cache — caches the generated SQL (not the final answer;
# data changes over time, so the answer is always re-computed by re-running
# the SQL and re-summarizing fresh rows). What's expensive and safe to
# reuse is the LLM figuring out WHICH tables are relevant and HOW to write
# the SQL — that logic depends on the schema, which barely ever changes,
# not on the data inside it.
#
# This is a simple exact/near-duplicate cache (same question text,
# normalized) — it will NOT catch differently-worded questions that mean
# the same thing (e.g. "how many BOXNM2 wagons" vs "how many wagons of
# type BOXNM2" normalize to two different strings and will both miss). A
# semantic cache (embedding-based similarity lookup against previously
# cached questions) would catch more repeats, at the cost of needing a
# small vector index and a tuned similarity threshold — worth adding later
# if this simple version isn't cutting enough LLM calls in practice.
# ---------------------------------------------------------------------------
_SQL_CACHE_TTL_SECONDS = 6 * 60 * 60  # schema-driven, so a long TTL is safe
_SQL_CACHE_MAX_ENTRIES = 500

_sql_cache: dict[str, dict] = {}
_sql_cache_order: list[str] = []  # tracks insertion order for FIFO eviction
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
    """
    Asks the LLM to decide whether `question` is answerable from the
    master-data tables, and if so, to write the SQL for it — rather than
    matching the question against a fixed set of hand-written patterns
    (which can only ever cover the cases someone thought to anticipate).
    Any count, list, filter, join across tables, date calculation, or
    cross-field summary is handled by the same mechanism, because the LLM
    is writing real SQL against the real, verified schema, not being
    routed through pre-defined categories.

    The schema text itself is built per-question from only the tables
    _select_relevant_tables() decides are relevant — see that function's
    docstring for why this matters as the table catalog grows.

    A cache is checked first (see _get_cached_plan) — a repeat or
    near-identical question skips BOTH the table-selection call and the
    SQL-writing call entirely.
    """
    if not question or not question.strip():
        return None

    cached_plan = _get_cached_plan(question)
    if cached_plan is not None:
        return cached_plan

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
    called out as not queryable — treat those as plain codes). When a
    question needs both deduping p_roams_wagon_master AND a JOIN/GROUP BY/
    aggregate, follow the required CTE-first pattern shown above exactly
    — do not combine DISTINCT ON with GROUP BY in the same SELECT level.
    Use unnest() for array columns when their contents matter, and
    ->>'key' for incharge_details fields. Never write more than one
    statement, and never write anything other than a SELECT (with an
    optional WITH prefix).

    For any comparison against a free-text or high-cardinality categorical
    column (state_name, district_name, org_desc, station_name, owner_name,
    wagon_description, rake_type, stock_type, location_type, etc.), always
    use case-insensitive matching (ILIKE, not =), since the underlying
    data may be stored in mixed, inconsistent, or slightly messy form.

    If none of the tables shown above can answer this question — for
    example if it's about an uploaded document/manual/procedure, or is
    unrelated to this data — set can_answer_with_sql to false and omit
    sql.
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": SQLQueryPlan,
            },
        )
        plan = json.loads(response.text)
    except Exception as e:
        logger.warning(f"SQL plan generation failed: {e}")
        return None

    # Only cache a plan that actually resolved to real SQL — caching a
    # "can't answer this" result would wrongly stick if the schema/catalog
    # is expanded later to cover it (a false negative is worse to cache
    # than a false positive, since a cache-miss just costs one extra call).
    if plan.get("can_answer_with_sql") and plan.get("sql"):
        _store_cached_plan(question, plan)

    return plan


def execute_sql_query(sql: str) -> list[dict]:
    cleaned = sql.strip().rstrip(";")
    with source_engine.connect() as conn:
        result = conn.execute(text(cleaned))
        columns = result.keys()
        # fetchmany caps the result regardless of what the query itself
        # specifies (e.g. no LIMIT, or a very large one).
        rows = result.fetchmany(MAX_RESULT_ROWS)
        return [dict(zip(columns, row)) for row in rows]


def summarize_sql_results(question: str, sql: str, rows: list[dict]) -> str:
    """
    Turns the raw SQL result into a natural-language answer. The model
    only sees the query and its actual returned rows — it isn't asked to
    recall or infer anything beyond what's in front of it, so the answer
    stays grounded in the real result set (including its true size, since
    MAX_RESULT_ROWS caps what's fetched but COUNT()-based questions return
    their exact total as a single row anyway).
    """
    if not rows:
        return "NOT_FOUND"

    rows_text = json.dumps(rows, default=str)

    prompt = f"""
    This SQL query was run against the railway master-data database to
    answer the user's question:

    {sql}

    It returned {len(rows)} row(s):
    {rows_text}

    Using ONLY this result, answer the question in clear natural language.

    IMPORTANT: rows were actually returned, so this question DOES have an
    answer — do not respond with "NOT_FOUND" just because some requested
    field is NULL/missing for some or all rows. In that case, still answer
    using whatever fields ARE populated, and explicitly say which
    requested field was not available (e.g. "state and district are not
    recorded for these owners"). Reserve "NOT_FOUND" strictly for cases
    where these specific rows are about a completely different entity than
    what was asked (e.g. the question asks for X but every returned row is
    about Y, with no X-related column at all) — never for missing/NULL
    values within an otherwise relevant result.

    Question: {question}
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"temperature": 0},
        )
        return (response.text or "").strip()
    except Exception as e:
        logger.warning(f"summarize_sql_results failed: {e}")
        return "NOT_FOUND"


def answer_master_data_query(question: str) -> Optional[dict]:
    """
    Full pipeline: route -> classify -> validate -> execute -> summarize.
    Returns None at any point the question isn't a master-data SQL
    question, the generated SQL fails validation, execution errors out, or
    the result doesn't actually answer the question — in every such case
    the caller should fall back to normal chunk-based retrieval instead of
    surfacing a broken structured-query attempt.
    """
    plan = generate_sql_query(question)
    if not plan or not plan.get("can_answer_with_sql") or not plan.get("sql"):
        return None

    sql = plan["sql"]
    # Re-validated even for a cache hit — cheap (no LLM call), and guards
    # against a future code change to _ALLOWED_TABLES making an old cached
    # query unsafe/out of scope.
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