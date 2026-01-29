import csv
import os
import requests
from io import StringIO

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import psycopg
from sentence_transformers import SentenceTransformer

print("🚀 UrbanEats starting up...")

API_PREFIX = "/api/v1"
app = FastAPI()

# =====================================================
# ✅ CORS (REQUIRED FOR FRONTEND)
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Lazy-loaded embedding model
# =====================================================
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("🧠 Loading embedding model...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model loaded")
    return _embedding_model

# =====================================================
# Database
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
# Health
# =====================================================
@app.get(f"{API_PREFIX}/health")
def health():
    return {"status": "ok"}

@app.get(f"{API_PREFIX}/ready")
def ready():
    return {"status": "ready"}

# =====================================================
# Products
# =====================================================
@app.get(f"{API_PREFIX}/products")
def get_products(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50)
):
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

            cur.execute("SELECT COUNT(*) FROM products")
            total = cur.fetchone()[0]

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "products": rows
    }

# =====================================================
# Full-text search
# =====================================================
@app.get(f"{API_PREFIX}/search")
def search_products(q: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, description, price
                FROM products
                WHERE search_vector @@ plainto_tsquery('english', %s)
                ORDER BY ts_rank(search_vector, plainto_tsquery('english', %s)) DESC
                """,
                (q, q)
            )
            rows = cur.fetchall()

    return {"query": q, "results": rows}

# =====================================================
# Google Sheet import
# =====================================================
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1iSyV2KTMxpyy4ylNn1DY5GMPSi12qruQt6SK85mK_Hg/export?format=csv"

@app.post(f"{API_PREFIX}/admin/import-sheet")
def import_sheet():
    response = requests.get(SHEET_CSV_URL)
    reader = csv.DictReader(StringIO(response.text))

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for row in reader:
                cur.execute(
                    """
                    INSERT INTO products (name, description, price)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (row["name"], row["description"], row["price"])
                )
        conn.commit()

    return {"status": "imported"}

# =====================================================
# Embeddings
# =====================================================
def generate_embedding(text: str):
    return get_embedding_model().encode(text).tolist()

@app.post(f"{API_PREFIX}/admin/embed-products")
def embed_products():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, description FROM products")
            rows = cur.fetchall()

            for pid, name, desc in rows:
                emb = generate_embedding(f"{name}. {desc}")
                cur.execute(
                    "UPDATE products SET embedding = %s WHERE id = %s",
                    (emb, pid)
                )
        conn.commit()

    return {"count": len(rows)}

# =====================================================
# Semantic Search
# =====================================================
@app.get(f"{API_PREFIX}/semantic-search")
def semantic_search(q: str, limit: int = 5):
    query_embedding = generate_embedding(q)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, description, price,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM products
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
                LIMIT %s
                """,
                (query_embedding, limit)
            )
            rows = cur.fetchall()

    return {"query": q, "results": rows}

# =====================================================
# Hybrid Search (Frontend uses this)
# =====================================================
@app.get("/api/v1/hybrid-search")
def hybrid_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20)
):
    query_embedding = generate_embedding(q.lower())

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
WITH ranked AS (
    SELECT
        id,
        name,
        description,
        price,
        tags,
        ts_rank(search_vector, plainto_tsquery('english', %s)) AS text_rank,
        1 - (embedding <=> %s::vector) AS semantic_score,

        CASE
            WHEN %s = ANY(tags) THEN 0.3
            ELSE 0
        END AS tag_boost
    FROM products
)
SELECT DISTINCT ON (name)
    id,
    name,
    description,
    price,
    text_rank,
    semantic_score,
    (text_rank * 0.6 + semantic_score * 0.3 + tag_boost) AS final_score
FROM ranked
WHERE text_rank > 0 OR semantic_score > 0.2
ORDER BY name, final_score DESC
LIMIT %s;
                """,
                (q, query_embedding, q, limit)
            )
            rows = cur.fetchall()

    return {
        "query": q,
        "results": [
            {
                "id": r[0],
                "name": r[1],
                "description": r[2],
                "price": float(r[3]),
                "text_rank": float(r[4]),
                "semantic_score": float(r[5]),
                "final_score": float(r[6]),
            }
            for r in rows
        ]
    }

# =====================================================
# Run
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
