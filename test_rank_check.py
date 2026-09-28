# test_rank_check.py — is file ko backend folder mein banao aur chalao:
# python test_rank_check.py

from app.database import SessionLocal
from app.models import Chunk
from app.services.embedding_service import generate_query_embedding

db = SessionLocal()

query = "how to login into irfmm"
query_embedding = generate_query_embedding(query)

# Saare chunks ko unki cosine_distance ke hisaab se sorted nikaalo
# (koi limit nahi, taaki poori ranking dikhe)
all_ranked = (
    db.query(Chunk)
    .order_by(Chunk.embedding.cosine_distance(query_embedding))
    .all()
)

# Us specific chunk (id=2, jisme "login to IRFMM" hai) ka rank dhoondo
for rank, chunk in enumerate(all_ranked, start=1):
    if "login to IRFMM" in chunk.text:
        print(f"MATCH FOUND — rank: {rank} out of {len(all_ranked)}")
        print(f"chunk id: {chunk.id}, page: {chunk.page_number}")
        print(f"text: {chunk.text[:100]}")
        break
else:
    print("Chunk not found in ranked results at all — unexpected!")

db.close()