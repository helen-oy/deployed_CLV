#!/usr/bin/env python3
"""
Script to run the CLV prediction API server.
"""

import uvicorn
import argparse
import sys
from pathlib import Path

# Add project root to path so package imports work
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Run the FastAPI server."""
    parser = argparse.ArgumentParser(description='Run CLV Prediction API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')

    args = parser.parse_args()

    print(f"Starting CLV Prediction API on {args.host}:{args.port}")

    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == '__main__':
    main()
