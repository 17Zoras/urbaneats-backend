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
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("🧠 Loading embedding model (lazy)...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model loaded")
    return embedding_model

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
# Health check
# =====================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "UrbanEats backend running"
    }

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
# Products API (pagination)
# =====================================================
@app.get("/products")
def get_products(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50)
):
    try:
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

    except Exception as e:
        return {"error": "db error", "detail": str(e)}

# =====================================================
# Full-Text Search (Chapter 6)
# =====================================================
@app.get("/search")
def search_products(q: str = Query(..., min_length=1)):
    try:
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

    except Exception as e:
        return {"error": "search failed", "detail": str(e)}

# =====================================================
# Google Sheet Import (Admin)
# =====================================================
SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1iSyV2KTMxpyy4ylNn1DY5GMPSi12qruQt6SK85mK_Hg"
    "/export?format=csv"
)

def import_products_from_sheet():
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
                    (
                        row["name"],
                        row["description"],
                        row["price"]
                    )
                )
        conn.commit()

@app.post("/admin/import-sheet")
def import_sheet():
    try:
        import_products_from_sheet()
        return {"status": "imported"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# Embedding helpers (Chapter 7)
# =====================================================
def generate_embedding(text: str):
    model = get_embedding_model()
    vector = model.encode(text)
    return vector.tolist()

@app.post("/admin/embed-products")
def embed_products():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, name, description FROM products;")
                rows = cur.fetchall()

                for product_id, name, description in rows:
                    text = f"{name}. {description}"
                    embedding = generate_embedding(text)

                    cur.execute(
                        """
                        UPDATE products
                        SET embedding = %s
                        WHERE id = %s
                        """,
                        (embedding, product_id)
                    )

        conn.commit()
        return {"status": "embeddings generated", "count": len(rows)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# Semantic Search (pgvector cosine similarity)
# =====================================================
@app.get("/semantic-search")
def semantic_search(q: str = Query(..., min_length=1), limit: int = 5):
    try:
        query_embedding = generate_embedding(q)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, description, price,
                           1 - (embedding <=> %s) AS similarity
                    FROM products
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, limit)
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
# Local run (ignored by Cloud Run)
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )
