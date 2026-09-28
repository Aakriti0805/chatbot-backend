import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import document_routes
from app.routers import query_routes
from app.routers import ocr_routes
from app.routers import master_data_routes

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Railway Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document_routes.router)
app.include_router(query_routes.router)
app.include_router(ocr_routes.router)
app.include_router(master_data_routes.router)

@app.get("/")
def root():
    return {"message": "Chatbot backend is running"}