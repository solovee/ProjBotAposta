web: gunicorn src.main:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT 
worker: python src/main.py 