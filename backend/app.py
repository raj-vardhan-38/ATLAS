# ATLAS FastAPI Backend
# To run: uvicorn app:app --host 0.0.0.0 --port $PORT
import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from predict import run_analysis

app = FastAPI(title="ATLAS API")

# === CORS ===
# Replace these with your real frontend URL(s) in production:
origins = [
    "https://www.atlassystem.live",       # your Cloudflare domain
    "http://atlassystem.live",            # HTTP version
    "https://atlas-ca71.onrender.com",    # your Render default domain
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # set to ["*"] for quick local testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Health check ===
@app.get("/healthz")
def health():
    return {"status": "ok"}

# === Root ===
@app.get("/")
def root():
    return {"status": "Backend is running!"}

# === Run analysis endpoint ===
@app.post("/run_analysis")
async def analyze(file: UploadFile = None, data: str = Form(None)):
    try:
        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name
            result = run_analysis(tmp_path)
            os.remove(tmp_path)
            return result

        if data:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta", mode="w") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            result = run_analysis(tmp_path)
            os.remove(tmp_path)
            return result

        return {"status": "error", "message": "No input provided."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# === Run uvicorn server when executed with 'python app.py' ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
