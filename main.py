import csv
import os
import time
import requests
from io import StringIO
from collections import defaultdict

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import JSONResponse
import psycopg
from sentence_transformers import SentenceTransformer

print("🚀 UrbanEats starting up...")

app = FastAPI(title="UrbanEats API", version="1.0")

# =====================================================
# API VERSIONING
# =====================================================
API_PREFIX = "/api/v1"

# =====================================================
# RATE LIMITING (20 req / min / IP)
# =====================================================
RATE_LIMIT = 20
WINDOW = 60
requests_log = defaultdict(list)

@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    ip = request.client.host
    now = time.time()

    requests_log[ip] = [t for t in requests_log[ip] if now - t < WINDOW]

    if len(requests_log[ip]) >= RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limit", "message": "Too many requests"}
        )

    requests_log[ip].append(now)
    return await call_next(request)

# =====================================================
# STANDARD ERROR FORMAT
# =====================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "request_error", "message": exc.detail},
    )

# =====================================================
# LAZY EMBEDDING MODEL
# =====================================================
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("🧠 Loading embedding model...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

# =====================================================
# DATABASE
# =====================================================
def get_db_connection():
    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME"),
        port=5432
    )

# =====================================================
# HEALTH + READINESS
# =====================================================
@app.get(f"{API_PREFIX}/health")
def health():
    return {"status": "ok"}

@app.get(f"{API_PREFIX}/ready")
def ready():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.execute("SELECT extname FROM pg_extension WHERE extname='vector';")
                if not cur.fetchone():
                    raise Exception("pgvector missing")
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# PRODUCTS
# =====================================================
@app.get(f"{API_PREFIX}/products")
def get_products(page: int = 1, limit: int = 10):
    offset = (page - 1) * limit
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, description, price
                FROM products
                ORDER BY id
                LIMIT %s OFFSET %s
                """,
                (limit, offset)
            )
            rows = cur.fetchall()
    return {"products": rows}

# =====================================================
# SEARCH
# =====================================================
@app.get(f"{API_PREFIX}/search")
def search(q: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, description, price
                FROM products
                WHERE search_vector @@ plainto_tsquery('english', %s)
                """,
                (q,)
            )
            rows = cur.fetchall()
    return {"query": q, "results": rows}

# =====================================================
# EMBEDDINGS
# =====================================================
def generate_embedding(text: str):
    return get_embedding_model().encode(text).tolist()

@app.post(f"{API_PREFIX}/admin/embed-products")
def embed_products():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, description FROM products;")
            for pid, name, desc in cur.fetchall():
                cur.execute(
                    "UPDATE products SET embedding=%s WHERE id=%s",
                    (generate_embedding(f"{name}. {desc}"), pid)
                )
        conn.commit()
    return {"status": "done"}

# =====================================================
# SEMANTIC SEARCH
# =====================================================
@app.get(f"{API_PREFIX}/semantic-search")
def semantic_search(q: str, limit: int = 5):
    emb = generate_embedding(q)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, description, price,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM products
                WHERE embedding IS NOT NULL
                  AND 1 - (embedding <=> %s::vector) >= 0.45
                ORDER BY similarity DESC
                LIMIT %s
                """,
                (emb, emb, limit)
            )
            rows = cur.fetchall()
    return {"query": q, "results": rows}

# =====================================================
# HYBRID SEARCH
# =====================================================
@app.get(f"{API_PREFIX}/hybrid-search")
def hybrid_search(q: str, limit: int = 5):
    emb = generate_embedding(q)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, description, price,
                       ts_rank(search_vector, plainto_tsquery('english', %s)),
                       1 - (embedding <=> %s::vector)
                FROM products
                ORDER BY
                  (ts_rank(search_vector, plainto_tsquery('english', %s))*0.6 +
                   (1 - (embedding <=> %s::vector))*0.4) DESC
                LIMIT %s
                """,
                (q, emb, q, emb, limit)
            )
            rows = cur.fetchall()
    return {"query": q, "results": rows}

# =====================================================
# LOCAL RUN
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
