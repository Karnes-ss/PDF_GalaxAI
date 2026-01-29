import os

import uvicorn

import main


def main_entry() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(main.app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main_entry()