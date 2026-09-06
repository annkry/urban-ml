#!/usr/bin/env bash

set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
: "${DATABASE_URL:?set DATABASE_URL (the Neon connection string)}"

REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-urban-ml-ingest}"
SCHEDULER_JOB="${SCHEDULER_JOB:-urban-ml-ingest-5min}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-urban-ml-scheduler}"
SYSTEM_ID="${SYSTEM_ID:-bike_share_toronto}"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}"

echo "==> Enabling APIs"
gcloud services enable run.googleapis.com cloudscheduler.googleapis.com \
  cloudbuild.googleapis.com --project "${PROJECT_ID}"

echo "==> Building image"
gcloud builds submit --tag "${IMAGE}" --project "${PROJECT_ID}" .

echo "==> Deploying service (private: only the scheduler may invoke it)"
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --no-allow-unauthenticated \
  --command uvicorn \
  --args "urban_ml.api.ingest_app:app,--host,0.0.0.0,--port,8080" \
  --port 8080 \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 1 \
  --timeout 120 \
  --set-env-vars "DATABASE_URL=${DATABASE_URL},SYSTEM_ID=${SYSTEM_ID},APP_ENV=production"

SERVICE_URL="$(gcloud run services describe "${SERVICE}" \
  --region "${REGION}" --project "${PROJECT_ID}" --format 'value(status.url)')"
echo "==> Service at ${SERVICE_URL}"

echo "==> Service account for the scheduler"
gcloud iam service-accounts describe \
  "${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --project "${PROJECT_ID}" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "${SERVICE_ACCOUNT}" \
    --display-name "Invokes urban-ml ingestion" --project "${PROJECT_ID}"

gcloud run services add-iam-policy-binding "${SERVICE}" \
  --region "${REGION}" --project "${PROJECT_ID}" \
  --member "serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role roles/run.invoker >/dev/null

echo "==> Scheduler job, every 5 minutes"
SCHEDULER_ARGS=(
  --location "${REGION}"
  --project "${PROJECT_ID}"
  --schedule "*/5 * * * *"
  --uri "${SERVICE_URL}/ingest"
  --http-method POST
  --oidc-service-account-email "${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"
  --oidc-token-audience "${SERVICE_URL}"
  --attempt-deadline 120s
  --max-retry-attempts 1
)
if gcloud scheduler jobs describe "${SCHEDULER_JOB}" \
     --location "${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud scheduler jobs update http "${SCHEDULER_JOB}" "${SCHEDULER_ARGS[@]}"
else
  gcloud scheduler jobs create http "${SCHEDULER_JOB}" "${SCHEDULER_ARGS[@]}"
fi

echo
echo "Done. Verify with:"
echo "  gcloud scheduler jobs run ${SCHEDULER_JOB} --location ${REGION} --project ${PROJECT_ID}"
echo "  gcloud run services logs read ${SERVICE} --region ${REGION} --project ${PROJECT_ID} --limit 20"
