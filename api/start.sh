#!/bin/bash
cd "$(api/ "api/start.sh")"
exec gunicorn main:app -k uvicorn.workers.UvicornWorker --bind=0.0.0.0:10000
