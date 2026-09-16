# Urban ML Platform

Short-term availability forecasting for public bike-share systems. Polls a
[GBFS](https://gbfs.org) feed (Toronto Bike Share by default), stores the
history, trains a LightGBM model on it, and serves per-station predictions
over HTTP.

## Pipeline

1. **Ingest.** Every 5 minutes, the GBFS station feeds are fetched, validated,
   and staged as Parquet snapshots in object storage, which is the only sink
   (`src/urban_ml/ingestion/`).
2. **Archive.** A daily job exports complete days from the staging bucket to a
   Hugging Face dataset repo, then trims what the Hub confirms
   (`src/urban_ml/archive/`). Toronto's archive is published as
   [ankry/toronto-bikeshare](https://huggingface.co/datasets/ankry/toronto-bikeshare)
   under ODbL and grows daily.
3. **Train.** Features are built from the archive (or from Postgres when
   `HF_DATASET_REPO` is unset) and a LightGBM regressor is tracked in MLflow;
   artifacts land in `MODEL_DIR` (`src/urban_ml/modeling/`).
4. **Serve.** The API loads the model at startup and reads the recent window
   straight from object storage, with no database on the request path
   (`src/urban_ml/api/main.py`).

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness; touches nothing external. 503 when no model is loaded. |
| `GET /ready` | Whether a prediction is actually servable: model, storage, data freshness and history depth. |
| `GET /predict/{station_id}` | Vehicles available `FORECAST_HORIZON_MINUTES` (default 120) ahead. |

The ingestion service is a separate app (`urban_ml.api.ingest_app`) exposing
`POST /ingest` to run one cycle. It is deployed privately so only the scheduler
can invoke it.

## Quickstart

```bash
cp .env.example .env
make install
make ingest-gbfs          # one ingestion cycle
make run                  # API on http://localhost:8000/docs
```

Ingestion and `/predict` both go through the bucket, so both need
`GCS_BUCKET` and Google credentials (`gcloud auth application-default login`
locally). Without a bucket the API still answers `/health`, but ingestion
refuses to start. `docker compose up` runs the ingestion loop and the API
together, mounting your application default credentials into the loop.

## Deployment

`scripts/deploy.sh [all|api|ingest|archive]` provisions and deploys the whole
stack on Google Cloud Run: two services, an archive job, Cloud Scheduler
triggers, Artifact Registry and Secret Manager entries:

```bash
PROJECT_ID=your-project HF_DATASET_REPO=you/dataset ./scripts/deploy.sh all
```

## Development

`make all` runs lint, format, typecheck, tests, pre-commit and dead-code checks.
Run `make` targets individually as needed; see the `Makefile` for the rest
(`train`, `archive`, `mlflow-ui`, `db-revision`).
