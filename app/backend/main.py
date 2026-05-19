from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..common.config import ensure_storage_dirs, get_config
from .api.health import router as health_router
from .api.search import router as search_router

cfg = get_config()
ensure_storage_dirs(cfg)

app = FastAPI(title="BetterDay v2", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve thumbnails dưới /thumbnails để frontend hiển thị
thumb_dir = Path(cfg["storage"]["thumbnail_dir"])
thumb_dir.mkdir(parents=True, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=str(thumb_dir)), name="thumbnails")

app.include_router(health_router)
app.include_router(search_router)


@app.get("/")
def root():
    return {
        "service": "BetterDay v2",
        "endpoints": ["/health", "/api/search", "/api/stats", "/thumbnails/{...}"],
    }
