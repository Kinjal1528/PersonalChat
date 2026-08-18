from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ag import agent
from utils.cache import check_cache, update_cache
import os

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Updated paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # /project
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

# Mount static folder (CSS + JS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def get_homepage():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

class MessageRequest(BaseModel):
    message: str

@app.post("/ask")
async def ask(request: MessageRequest):
    try:
        query = request.message
        cached_reply = check_cache(query)
        if cached_reply:
            return {"response": cached_reply, "source": "cache"}
        reply = agent.run(query)
        update_cache(query, reply)
        return {"response": reply, "source": "agent"}
    except Exception as e:
        return {"response": f"Agent error: {str(e)}", "source": "error"}
