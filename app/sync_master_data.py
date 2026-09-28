import os
import json
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.models import MasterDataChunk
from app.services.embedding_service import generate_embeddings


load_dotenv(override=True)

SOURCE_DATABASE_URL = os.getenv("SOURCE_DATABASE_URL")
DEST_DATABASE_URL = os.getenv("DATABASE_URL")

if not SOURCE_DATABASE_URL:
    raise ValueError("SOURCE_DATABASE_URL is not set")

if not DEST_DATABASE_URL:
    raise ValueError("DATABASE_URL is not set")


source_engine = create_engine(SOURCE_DATABASE_URL)


dest_engine = create_engine(
    DEST_DATABASE_URL,
    connect_args={"options": "-csearch_path=chatbot,public"},
)

DestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=dest_engine)

# Ek baar mein kitne texts embed karne hain — pehli sync pe hazaaron rows
# hote hain, isliye chhote batches mein karte hain taaki memory/time ka issue na aaye.
EMBEDDING_BATCH_SIZE = 256


def _clean(value) -> str:
    """None/empty values ko literal 'null' bana deta hai, taaki header ke
    saath value hamesha chunk mein dikhe (jaise 'mf_recv_rly - null')."""
    if value is None or str(value).strip() == "":
        return "null"
    return str(value).strip()


def _row_to_text_generic(row: dict, columns: list[str]) -> str:
    """
    Row ke saare columns ko raw 'column_name - value' format mein, comma
    se separated likhta hai — koi translation/paraphrasing nahi, taaki CSV
    ki exact terminology se bhi retrieval match ho sake.
    """
    return ", ".join(f"{col} - {_clean(row.get(col))}" for col in columns)


OWNER_MASTER_COLUMNS = [
    "owner_id", "owner_name", "numeric_code", "address", "type",
    "created_by", "created_on", "modified_by", "modified_on",
    "valid_flag", "rb_owner_id", "fois_owner_id", "platform", "org_slno",
]

WAGON_TYPE_MASTER_COLUMNS = [
    "wagon_type", "wagon_description", "wagon_class_group", "numerical_code",
    "tare_tons", "cc_payload_tons", "axle_load", "bearing_type", "coupling_type",
    "bogie_type", "brake_type", "first_poh_duration", "next_poh_duration",
    "first_roh_duration", "next_roh_duration", "codal_life", "gauge",
    "avg_no_of_wagon_in_rake", "wagon_type_display_name", "no_of_wheels",
    "drawing_no", "valid_flag", "mcdo_flag", "old_wagon_type", "created_by",
    "created_on", "being_manufactured", "rly_pvt_owned", "platform",
    "updated_by", "updated_on", "org_slno", "mcdo_wagon_type", "vehicle_type",
    "rake_type", "prd_flag", "stock_type", "fois_mutiplication_factor",
    "groupname", "cvdtype", "img_url", "identification_desc",
    "first_allot_letter_image_url", "sample_image_url", "commercial_notif_no",
    "commercial_notif_date", "sr_no_in_commercial_notif", "remarks",
    "commercial_letter_image_url", "subsequent_poh_cycle_roh_duraton",
    "ir_empty_speed", "ir_loaded_speed", "color_code_speed", "speed_consent",
    "dfcc_empty_speed", "dfcc_loaded_speed", "brake_type_design",
    "universal_roh_duration",
]

# incharge_details JSON column ko alag se handle karte hain (flatten karke),
# isliye ise generic list mein nahi, row_to_text_org_m() mein alag jodte hain.
ORG_M_COLUMNS = [
    "org_slno", "org_code", "org_type", "org_desc", "org_seq", "prnt_org_slno",
    "num_code", "pvt_rly", "created_by", "created_on", "modified_by",
    "modified_on", "valid_flag", "manufacturer", "edrishti_dk",
    "district_name", "state_name", "gis_coord", "gis_flag", "remarks",
    "depot_rb_2017", "depot_rb_2008", "station_name", "man_power_flag",
    "mf_recv_rly", "dispatch_for_flag", "location_type",
    "org_slno_category", "rs_type_attended", "maintenance_activities_available",
]

# Baaki adb_* columns (adb_opcode, adb_subject, etc.) jaan-boojh ke nahi liye —
# ye replication/CDC ke internal columns hain, embedding mein noise banayenge.
# Sirf adb_sequence aur adb_timestamp rakhe hain, kyunki ek wagon_id ke multiple
# versions hain aur inse pata chalta hai ki kaunsa version kab ka hai (latest kaunsa).
P_WAGON_MASTER_COLUMNS = [
    "wagon_id", "wagon_no", "record_created_person", "wagon_type", "po_number",
    "tare", "manufacturer_date", "original_cost", "mf_id", "warranty_end_date",
    "manufacturer_country", "diagram_number", "manufacturer_serial_number",
    "commission_date", "commission_station", "wagon_owner",
    "wagon_carrying_capacity", "scheme_name", "pipe_fitted", "wagon_status",
    "base_station", "organization_type", "valid_flag", "roh_stn", "roh_date",
    "poh_stn", "poh_date", "brake_type", "rfid1", "rfid2", "poh_return_date",
    "pvt_vehicle_no", "rb_wagon_id", "intrm_remarks", "org_slno",
    "pvt_owner_code", "wagon_usage", "lessor_flag", "agreement_wagon_pool",
    "wagon_no_old", "update_in_no_type_owner",
    "adb_sequence", "adb_timestamp",
]


def row_to_text_owner_master(row: dict) -> str:
    return _row_to_text_generic(row, OWNER_MASTER_COLUMNS)


def row_to_text_wagon_type_master(row: dict) -> str:
    return _row_to_text_generic(row, WAGON_TYPE_MASTER_COLUMNS)


def row_to_text_p_wagon_master(row: dict) -> str:
    return _row_to_text_generic(row, P_WAGON_MASTER_COLUMNS)


def row_to_text_org_m(row: dict) -> str:
    base_text = _row_to_text_generic(row, ORG_M_COLUMNS)

    # incharge_details JSON ko flatten karke uske andar ke keys ko literal
    # naam ke saath jodte hain (contactNo, inchargeName, etc.)
    incharge_fields = "null"
    raw_incharge = row.get("incharge_details")
    if raw_incharge:
        try:
            details = raw_incharge if isinstance(raw_incharge, dict) else json.loads(raw_incharge)
            address_obj = details.get("officePostalAddress") or {}
            incharge_fields = (
                f"inchargeName - {_clean(details.get('inchargeName'))}, "
                f"inchargeDesignation - {_clean(details.get('inchargeDesignation'))}, "
                f"contactNo - {_clean(details.get('contactNo'))}, "
                f"city - {_clean(address_obj.get('city'))}, "
                f"state - {_clean(address_obj.get('state'))}, "
                f"pincode - {_clean(address_obj.get('pincode'))}"
            )
        except Exception:
            incharge_fields = "null"

    return f"{base_text}, incharge_details - ({incharge_fields})"


MASTER_TABLE_CONFIGS = [
    {
        "source_table": "roams_owner_master",
        "primary_key": "owner_id",
        "query": f"""
            SELECT {", ".join(f'"{c}"' if c == "type" else c for c in OWNER_MASTER_COLUMNS)}
            FROM lb1adm.roams_owner_master
        """,
        "row_to_text": row_to_text_owner_master,
    },
    {
        "source_table": "roams_wagon_type_master",
        "primary_key": "wagon_type",
        "query": f"""
            SELECT {", ".join(WAGON_TYPE_MASTER_COLUMNS)}
            FROM lb1adm.roams_wagon_type_master
        """,
        "row_to_text": row_to_text_wagon_type_master,
    },
    {
        "source_table": "fmm_org_m",
        "primary_key": "org_slno",
        "query": f"""
            SELECT {", ".join(ORG_M_COLUMNS)}, incharge_details
            FROM lb1adm.fmm_org_m
        """,
        "row_to_text": row_to_text_org_m,
    },
    {
        "source_table": "p_roams_wagon_master",
        # Har row ka alag version chahiye (wagon_id same hota hai, baaki details
        # alag), isliye adb_sequence (source mein unique index wala column) ko
        # record_id bana rahe hain, taaki koi row skip na ho.
        "primary_key": "adb_sequence",
        "query": f"""
            SELECT {", ".join(P_WAGON_MASTER_COLUMNS)}
            FROM lb1adm.p_roams_wagon_master
            ORDER BY adb_sequence
        """,
        "row_to_text": row_to_text_p_wagon_master,
    },
]


def _fetch_source_rows(config: dict) -> list[dict]:
    with source_engine.connect() as conn:
        result = conn.execute(text(config["query"]))
        columns = result.keys()
        return [dict(zip(columns, row)) for row in result.fetchall()]


def sync_master_table(config: dict, db) -> dict:
    """
    Ek master table ke saare rows sync karta hai: naya row insert, badla
    hua row update (embedding regenerate karke), unchanged row skip.
    Har batch ke baad commit hota hai, isliye beech mein ruk jaye to dobara
    run karne pe wahin se resume hota hai.
    """
    source_table = config["source_table"]
    primary_key = config["primary_key"]

    rows = _fetch_source_rows(config)
    print(f"  Fetched {len(rows)} rows from {source_table}")

    # Same primary key wale rows ko dedupe karte hain (safety ke liye) — jo row
    # baad mein aati hai wahi rakhte hain. p_roams_wagon_master mein primary key
    # adb_sequence unique hai, isliye wahan koi row skip nahi hogi.
    unique_rows = {}
    for row in rows:
        record_id = str(row.get(primary_key))
        if not record_id or record_id == "None":
            continue
        unique_rows[record_id] = row

    duplicates_skipped = len(rows) - len(unique_rows)
    if duplicates_skipped:
        print(f"  Skipped {duplicates_skipped} duplicate/null-key rows in {source_table}")

    existing_chunks = {
        c.record_id: c
        for c in db.query(MasterDataChunk)
        .filter(MasterDataChunk.source_table == source_table)
        .all()
    }

    inserted = 0
    updated = 0
    unchanged = 0

    texts_to_embed = []
    chunks_needing_embedding = []

    for record_id, row in unique_rows.items():
        new_text = config["row_to_text"](row)
        existing = existing_chunks.get(record_id)

        if existing is None:
            new_chunk = MasterDataChunk(
                source_table=source_table,
                record_id=record_id,
                text=new_text,
            )
            # db.add yahan nahi karte — embedding ban jaane ke baad hi add karenge,
            # warna beech mein commit hone pe bina embedding wale rows save ho jayenge
            # aur agle run mein 'unchanged' maan liye jayenge.
            texts_to_embed.append(new_text)
            chunks_needing_embedding.append(new_chunk)
            inserted += 1
        elif existing.text != new_text or existing.embedding is None:
            # embedding None wale rows (pichhle adhoore run se) bhi dobara embed honge
            existing.text = new_text
            existing.embedding = None
            texts_to_embed.append(new_text)
            chunks_needing_embedding.append(existing)
            updated += 1
        else:
            unchanged += 1

    if texts_to_embed:
        total = len(texts_to_embed)
        for i in range(0, total, EMBEDDING_BATCH_SIZE):
            batch_texts = texts_to_embed[i:i + EMBEDDING_BATCH_SIZE]
            batch_chunks = chunks_needing_embedding[i:i + EMBEDDING_BATCH_SIZE]
            embeddings = generate_embeddings(batch_texts)
            for chunk, embedding in zip(batch_chunks, embeddings):
                chunk.embedding = embedding
                db.add(chunk)
            # Har batch ke baad commit, taaki beech mein ruk jaye to progress save rahe
            db.commit()
            print(f"  Embedded + saved {min(i + EMBEDDING_BATCH_SIZE, total)}/{total}")
    else:
        db.commit()

    print(f"  {source_table}: inserted={inserted}, updated={updated}, unchanged={unchanged}")

    return {"source_table": source_table, "inserted": inserted, "updated": updated, "unchanged": unchanged}


def run_master_data_sync() -> dict:
    """
    Saare configured master tables ko sync karta hai. CLI se ya FastAPI
    background task se, dono jagah se reusable.
    """
    print("=" * 70)
    print("MASTER DATA SYNC -> chatbot.master_data_chunks")
    print("=" * 70)

    db = DestSessionLocal()
    results = []

    try:
        for config in MASTER_TABLE_CONFIGS:
            print(f"\nSyncing {config['source_table']}...")
            try:
                results.append(sync_master_table(config, db))
            except Exception as e:
                print(f"  -> Failed to sync {config['source_table']}: {e}")
                db.rollback()

        print("\n" + "=" * 70)
        print("MASTER DATA SYNC COMPLETED")
        print("=" * 70)

        return {"tables": results}

    finally:
        db.close()


if __name__ == "__main__":
    run_master_data_sync()