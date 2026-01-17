from fastapi import FastAPI
from app.routers import health

app = FastAPI(title="UrbanEats Backend")

app.include_router(health.router)

@app.get("/")
def root():
    return {"message": "UrbanEats backend is running"}
