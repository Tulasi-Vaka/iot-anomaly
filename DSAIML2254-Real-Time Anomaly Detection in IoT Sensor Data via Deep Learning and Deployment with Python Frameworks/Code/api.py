# Vercel entrypoint - imports the FastAPI app from api/index.py
from api.index import app

# Export the app for Vercel
__all__ = ["app"]
