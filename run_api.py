import os
import sys
import uvicorn

def main() -> None:
    host = os.getenv("INTELLPULSE_HOST", "127.0.0.1")
    port = int(os.getenv("INTELLPULSE_PORT", "8000"))
    reload_ = os.getenv("INTELLPULSE_RELOAD", "true").lower() in ("1", "true", "yes", "y")

    app_path = os.getenv("INTELLPULSE_APP", "src.api.app:app")
    docs_url = f"http://{host}:{port}/docs"

    print("\n🚀 Starting Intellpulse API")
    print(f"   App: {app_path}")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Reload: {reload_}")
    print(f"   Docs: {docs_url}\n")

    try:
        uvicorn.run(app_path, host=host, port=port, reload=reload_, log_level="info")
    except Exception as e:
        print(f"❌ Failed to start Uvicorn: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    main()
