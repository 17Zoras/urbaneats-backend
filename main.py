import csv
import os
import requests
from io import StringIO

from fastapi import FastAPI, Query, Header, HTTPException
import psycopg

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



print("🚀 UrbanEats starting up...")

app = FastAPI()



def generate_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def embed_all_products():
    print("🧠 Generating embeddings for products...")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # fetch products that don't have embeddings yet
            cur.execute("""
                SELECT id, name, description
                FROM products
                WHERE embedding IS NULL
            """)
            rows = cur.fetchall()

            for product_id, name, description in rows:
                text = f"{name} {description}"

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

    print("✅ Embeddings stored successfully")



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
# Admin security (Chapter 9)
# =====================================================
def verify_admin_key(x_admin_key: str | None):
    expected = os.getenv("ADMIN_KEY")
    if not expected or x_admin_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

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
# Search API (Postgres Full-Text Search – Chapter 6)
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
# Google Sheet Import (ADMIN ONLY)
# =====================================================
SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1iSyV2KTMxpyy4ylNn1DY5GMPSi12qruQt6SK85mK_Hg"
    "/export?format=csv"
)

def import_products_from_sheet():
    print("📥 Importing products from Google Sheet...")

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
                        row["price"],
                    )
                )
        conn.commit()

    print("✅ Product import completed")

# =====================================================
# Admin endpoint (SECURED)
# =====================================================
@app.post("/admin/import-sheet")
def import_sheet(x_admin_key: str | None = Header(default=None)):
    verify_admin_key(x_admin_key)
    import_products_from_sheet()
    return {"status": "imported"}



@app.post("/admin/embed-products")
def embed_products():
    try:
        embed_all_products()
        return {"status": "embeddings generated"}
    except Exception as e:
        return {"status": "failed", "detail": str(e)}




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
