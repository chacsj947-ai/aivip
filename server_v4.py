# server_v4.py
import os
import json
import httpx
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
import sqlite3
from passlib.context import CryptContext
import jwt
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- CẤU HÌNH & KẾT NỐI ---
SECRET_KEY = "wormgpt_v52_dark_multi_secret"
ALGORITHM = "HS256"
DB_NAME = "wormgpt_users_v4.db"

# API KEYS (Bạn nên đặt biến môi trường cho mỗi key)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROK_API_KEY = os.getenv("XAI_API_KEY") # xAI API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI(title="WormGPT v5.2 Multi-Model", version="4.0.0")

# --- MODEL REGISTRY (BẢN TRA CỨU MÔ HÌNH) ---
# Key: Model ID, Value: { provider, url, headers_builder }
MODEL_REGISTRY = {
    "gpt-4o": {
        "provider": "openai",
        "url": "https://api.openai.com/v1/chat/completions",
        "key": OPENAI_API_KEY
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "url": "https://api.openai.com/v1/chat/completions",
        "key": OPENAI_API_KEY
    },
    "claude-3-5-sonnet-20241022": {
        "provider": "anthropic",
        "url": "https://api.anthropic.com/v1/messages",
        "key": ANTHROPIC_API_KEY
    },
    "claude-3-haiku-20240307": {
        "provider": "anthropic",
        "url": "https://api.anthropic.com/v1/messages",
        "key": ANTHROPIC_API_KEY
    },
    "grok-beta": {
        "provider": "xai",
        "url": "https://api.x.ai/v1/chat/completions",
        "key": GROK_API_KEY
    },
    "gemini-1.5-pro": {
        "provider": "google",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
        "key": GOOGLE_API_KEY
    }
}

# --- DATABASE & SECURITY ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            level INTEGER DEFAULT 1
        )
    ''')
    if cursor.execute("SELECT count(*) FROM users WHERE username='admin'").fetchone()[0] == 0:
        cursor.execute("INSERT INTO users VALUES (1, 'admin', ?, 5)", (pwd_context.hash("admin123"),))
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

# --- HELPER FUNCTIONS ---
def get_model_config(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found in registry.")
    config = MODEL_REGISTRY[model_name]
    if not config.get("key"):
        raise HTTPException(status_code=500, detail=f"API Key for {config['provider']} not configured.")
    return config

def extract_content_delta(data: dict) -> str:
    """Tự động trích xuất nội dung dựa trên cấu trúc API khác nhau"""
    # OpenAI / Grok style
    if "choices" in data and len(data["choices"]) > 0:
        delta = data["choices"][0].get("delta", {})
        if "content" in delta:
            return delta["content"]
    
    # Anthropic style
    if "content" in data:
        blocks = data["content"]
        if blocks and isinstance(blocks, list):
            for block in blocks:
                if block.get("type") == "text" and "text" in block:
                    return block["text"]
    
    return ""

# --- AUTH LOGIC (Giữ nguyên từ v3) ---
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=60*24*7)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except:
        raise HTTPException(status_code=401, detail="Invalid Token")
    
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {"username": user["username"], "level": user["level"]}

# --- ROUTES ---
@app.on_event("startup")
async def startup():
    init_db()

@app.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (form_data.username,)).fetchone()
    conn.close()
    if not user or not pwd_context.verify(form_data.password, user["password"]):
        raise HTTPException(400, detail="Invalid credentials")
    
    token = create_access_token({"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/models")
async def list_models():
    """Trả về danh sách các mô hình đang hỗ trợ"""
    return {
        "models": [
            {"id": k, "provider": v["provider"], "name": f"{k} ({v['provider'].upper()})"}
            for k, v in MODEL_REGISTRY.items()
        ]
    }

@app.post("/chat")
async def chat_endpoint(req: dict, token: str = Depends(oauth2_scheme)):
    user = get_current_user(token)
    model_name = req.get("model", "gpt-4o")
    
    # Lấy cấu hình API
    config = get_model_config(model_name)
    api_key = config["key"]
    url = config["url"]
    
    # Xây dựng payload riêng cho từng provider
    if config["provider"] == "anthropic":
        payload = {
            "model": model_name,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": req["message"]}]
        }
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    elif config["provider"] == "google":
        payload = {
            "contents": [{"parts": [{"text": req["message"]}] }]
        }
        headers = {"x-goog-api-key": api_key}
    else:
        # OpenAI / Grok / Generic
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": req["message"]}],
            "stream": True
        }
        headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="API Error")
            
            async def stream_gen():
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        try:
                            data = json.loads(data_str)
                            content = extract_content_delta(data)
                            if content:
                                yield f"data: {json.dumps({'content': content})}\n\n"
                        except json.JSONDecodeError:
                            continue
            return StreamingResponse(stream_gen(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)