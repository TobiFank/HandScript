# backend/run.py
import sys
import uvicorn
from app.config import settings

def main():
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )
    except Exception as e:
        print(f"Error starting the server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()