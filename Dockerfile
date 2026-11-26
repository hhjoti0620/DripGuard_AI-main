FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 dripuser && mkdir -p /data && chown -R dripuser:dripuser /app /data

COPY --chown=dripuser:dripuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=dripuser:dripuser . .

USER dripuser

ENV DATABASE_URL="sqlite:////data/dripguard.db" \
    PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]