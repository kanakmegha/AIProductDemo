"""FastAPI application entry point for the AI Product Demo Studio backend."""
import logging

from dotenv import load_dotenv
from fastapi import FastAPI


from backend.intelligence.router import router as intelligence_router

load_dotenv()

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AI Product Demo Studio")
app.include_router(intelligence_router)


@app.get("/")
async def root():
    """Return a basic liveness message for the root path."""
    return {"service": "AI Product Demo Studio", "status": "running"}
