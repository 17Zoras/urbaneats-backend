import csv
import os
import requests
from io import StringIO

from fastapi import FastAPI, Query, HTTPException
import psycopg
from sentence_transformers import SentenceTransformer

print("🚀 UrbanEats starting up...")

app = FastAPI()

# =====================================================
# Lazy-loaded embedding model (Cloud Run safe)
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
# Database connection
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
@app.get("/health")
def health():
    return {"status": "ok", "service": "UrbanEats backend running"}

# =====================================================
# DB test
# =====================================================
@app.get("/db-test")
def db_test():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
        return {"db": "connected"}
    except Exception as e:
        return {"db": "error", "detail": str(e)}

# =====================================================
# Products (pagination)
# =====================================================
@app.get("/products")
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

            cur.execute("SELECT COUNT(*) FROM products;")
            total = cur.fetchone()[0]

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "products": [
            {
                "id": r[0],
                "name": r[1],
                "description": r[2],
                "price": float(r[3])
            }
            for r in rows
        ]
    }

# =====================================================
# Full-text search
# =====================================================
@app.get("/search")
def search_products(q: str = Query(..., min_length=1)):
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

    return {
        "query": q,
        "results": [
            {
                "id": r[0],
                "name": r[1],
                "description": r[2],
                "price": float(r[3])
            }
            for r in rows
        ]
    }

# =====================================================
# Google Sheet import (admin)
# =====================================================
SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1iSyV2KTMxpyy4ylNn1DY5GMPSi12qruQt6SK85mK_Hg"
    "/export?format=csv"
)

@app.post("/admin/import-sheet")
def import_sheet():
    response = requests.get(SHEET_CSV_URL, timeout=30)
    response.raise_for_status()

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
# Embedding helpers
# =====================================================
def generate_embedding(text: str):
    model = get_embedding_model()
    return model.encode(text).tolist()

# =====================================================
# Admin: generate embeddings
# =====================================================
@app.post("/admin/embed-products")
def embed_products():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, description FROM products;")
            rows = cur.fetchall()

            for pid, name, desc in rows:
                emb = generate_embedding(f"{name}. {desc}")
                cur.execute(
                    "UPDATE products SET embedding = %s WHERE id = %s",
                    (emb, pid)
                )

        conn.commit()

    return {"status": "embeddings generated", "count": len(rows)}

# =====================================================
# ✅ SEMANTIC SEARCH (FIXED + CLEAN)
# =====================================================
@app.get("/semantic-search")
def semantic_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20)
):
    try:
        query_embedding = generate_embedding(q)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH scored AS (
                        SELECT
                            id,
                            name,
                            description,
                            price,
                            1 - (embedding <=> %s::vector) AS similarity
                        FROM products
                        WHERE embedding IS NOT NULL
                    )
                    SELECT *
                    FROM scored
                    WHERE similarity >= 0.45
                    ORDER BY similarity DESC
                    LIMIT %s;
                    """,
                    (query_embedding, limit)
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
                    "similarity": float(r[4])
                }
                for r in rows
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# 🔀 Hybrid Search (Text + Semantic) — Chapter 8
# =====================================================

@app.get("/hybrid-search")
def hybrid_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20)
):
    try:
        query_embedding = generate_embedding(q)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
SELECT DISTINCT ON (name)
    id,
    name,
    description,
    price,
    ts_rank(search_vector, plainto_tsquery('english', %s)) AS text_rank,
    1 - (embedding <=> %s::vector) AS semantic_score
FROM products
WHERE
    search_vector @@ plainto_tsquery('english', %s)
    OR (1 - (embedding <=> %s::vector)) > 0.35
ORDER BY
    name,
    (ts_rank(search_vector, plainto_tsquery('english', %s)) * 0.6
     + (1 - (embedding <=> %s::vector)) * 0.4) DESC
LIMIT %s;
                    """,
                    (
                        q,
                        query_embedding,
                        q,
                        query_embedding,
                        q,
                        query_embedding,
                        limit
                    )
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
                    "text_rank": float(r[4]) if r[4] else 0.0,
                    "semantic_score": float(r[5]),
                    "category_boost": float(r[6]),
                }
                for r in rows
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# =====================================================
# 🏷️ Filter by Tag — Chapter 9
# =====================================================
@app.get("/filter-by-tag")
def filter_by_tag(
    tag: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50)
):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, description, price, tags
                FROM products
                WHERE %s = ANY(tags)
                LIMIT %s
                """,
                (tag, limit)
            )
            rows = cur.fetchall()

    return {
        "tag": tag,
        "results": [
            {
                "id": r[0],
                "name": r[1],
                "description": r[2],
                "price": float(r[3]),
                "tags": r[4]
            }
            for r in rows
        ]
    }
# =====================================================
# Filter by category
# =====================================================
@app.get("/filter-by-category")
def filter_by_category(category: str = Query(..., min_length=1)):
    category = category.lower()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, description, price, tags, category
                FROM products
                WHERE category = %s
                ORDER BY id
                """,
                (category,)
            )
            rows = cur.fetchall()

    return {
        "category": category,
        "results": [
            {
                "id": r[0],
                "name": r[1],
                "description": r[2],
                "price": float(r[3]),
                "tags": r[4],
                "category": r[5]
            }
            for r in rows
        ]
    }


# =====================================================
# Local run (Cloud Run ignores this)
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )
