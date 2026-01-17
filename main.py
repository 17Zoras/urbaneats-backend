from fastapi import FastAPI
from app.routers import health
from app.config import APP_NAME

app = FastAPI(title=APP_NAME)

app.include_router(health.router)

@app.get("/")
def root():
    return {"message": f"{APP_NAME} is running"}
from app.db import engine

@app.get("/db-check")
def db_check():
    with engine.connect() as conn:
        return {"db": "connected"}
