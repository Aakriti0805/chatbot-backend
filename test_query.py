from app.database import SessionLocal
from app.services.query_services import answer_query

query = input("Enter your question: ")

db = SessionLocal()
result = answer_query(query, db)

print("\nANSWER:")
print(result["answer"])
print(f"\n(Found in round {result['rounds_used']} of search)")
print("\nSOURCES:")
for source in result["sources"]:
    print(f"- document_id={source['document_id']}, page={source['page_number']}")

db.close()