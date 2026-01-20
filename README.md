# 🍔 UrbanEats Backend

UrbanEats is a cloud-native, AI-ready food discovery backend built using **FastAPI**, **PostgreSQL**, and **Google Cloud Run**.  
This repository represents a **12-chapter full-stack + AI system**, with **Chapters 1–6 fully completed**.

---

## 🌍 Live Public API
Base URL:
```
https://urbaneats-backend-124394744291.asia-south1.run.app
```

This backend is **publicly accessible** and deployed on **Google Cloud Run**.

---

## 🧰 Tech Stack
- **FastAPI** – High-performance Python backend
- **PostgreSQL (Cloud SQL)** – Relational database
- **psycopg** – PostgreSQL driver
- **Docker** – Containerization
- **Google Cloud Run** – Serverless hosting
- **GitHub** – Version control

---

## ✅ Completed Chapters

### 📘 Chapter 1 – Project Setup
- FastAPI project initialization
- Environment-based configuration

### 📘 Chapter 2 – Backend Foundation
- Health check endpoint
- Database connectivity testing

### 📘 Chapter 3 – Database Integration
- PostgreSQL schema design
- Products table creation

### 📘 Chapter 4 – Cloud Deployment
- Dockerfile setup
- Cloud Run deployment
- Public service URL

### 📘 Chapter 5 – Data Ingestion
- Google Sheets → PostgreSQL import
- Safe admin-triggered import logic

### 📘 Chapter 6 – Search & Pagination
- Paginated products API
- PostgreSQL full-text search using `tsvector`
- Ranked search results
- Secure admin import endpoint

---

## 🔌 API Endpoints

### Health Check
```
GET /health
```

### Database Test
```
GET /db-test
```

### Products (Pagination)
```
GET /products?page=1&limit=10
```

### Full-Text Search
```
GET /search?q=burger
```

### Admin Import (POST only)
```
POST /admin/import-sheet
```

---

## 🔐 Security & Configuration
- All database credentials are managed using **Cloud Run Secrets**
- Admin import endpoint is **POST-only**
- No sensitive data is hard-coded

---

## 🛠 Local Development

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## 🧭 Upcoming Chapters (Planned)

- **Chapter 7** – AI Embeddings (Semantic Search)
- **Chapter 8** – User Behavior Tracking
- **Chapter 9** – Personalization Engine
- **Chapter 10** – AI Chatbot (RAG)
- **Chapter 11** – Notifications & Emails
- **Chapter 12** – Admin Analytics Dashboard
- **Frontend Integration** – Full-stack UI

---

## 👤 Author
**Zorawar Singh**  
Backend • Cloud • AI Engineering Project
