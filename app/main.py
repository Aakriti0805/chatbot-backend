from fastapi import FastAPI
from app.routers import document_routes
from app.routers import query_routes
from app.routers import ocr_routes

app = FastAPI(title="Railway Chatbot API")

app.include_router(document_routes.router)
app.include_router(query_routes.router)
app.include_router(ocr_routes.router)

@app.get("/")
def root():
    return {"message": "Chatbot backend is running"}