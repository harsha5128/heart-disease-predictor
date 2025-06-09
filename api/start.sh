#!/bin/bash
cd "$(dirname "$0")"
exec gunicorn main:app -k uvicorn.workers.UvicornWorker --bind=0.0.0.0:10000
