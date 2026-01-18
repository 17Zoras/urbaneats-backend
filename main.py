from fastapi import FastAPI
from app.db import init_db

app = FastAPI()

@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/")
def root():
    return {"message": "UrbanEats Backend is running"}
