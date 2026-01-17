from fastapi import FastAPI

app = FastAPI(title="UrbanEats Backend")

@app.get("/")
def root():
    return {"message": "UrbanEats backend is running"}
