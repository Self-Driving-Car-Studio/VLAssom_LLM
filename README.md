# VLAssom_LLM
명령 및 분기를 위한 LLM 구축

gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:3000 api.server:app
