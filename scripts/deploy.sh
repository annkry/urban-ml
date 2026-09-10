#!/usr/bin/env bash

set -euo pipefail

TARGET="${1:-all}"
case "${TARGET}" in
  all | api | ingest) ;;
  *)
    echo "usage: $(basename "$0") [all|api|ingest]" >&2
    exit 2
    ;;
esac

: "${PROJECT_ID:?set PROJECT_ID}"

REGION="${REGION:-us-east1}"
REPOSITORY="${REPOSITORY:-urban-ml}"
API_SERVICE="${API_SERVICE:-urban-ml-api}"
INGEST_SERVICE="${INGEST_SERVICE:-urban-ml-ingest}"
SCHEDULER_JOB="${SCHEDULER_JOB:-urban-ml-ingest-5min}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-urban-ml-scheduler}"
SECRET_NAME="${SECRET_NAME:-urban-ml-database-url}"
BUCKET="${BUCKET:-${PROJECT_ID}-urban-ml-staging}"

SYSTEM_ID="${SYSTEM_ID:-bike_share_toronto}"

if [[ "${ALLOW_UNAUTHENTICATED:-true}" == "true" ]]; then
  API_AUTH_FLAG="--allow-unauthenticated"
else
  API_AUTH_FLAG="--no-allow-unauthenticated"
fi

IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/urban-ml"
VERSION="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  VERSION="${VERSION}-dirty"
fi
IMAGE="${IMAGE_NAME}:${VERSION}"

echo "==> Target:  ${TARGET}"
echo "==> Image:   ${IMAGE}"
if [[ "${VERSION}" == *-dirty ]]; then
  echo "    WARNING: working tree has uncommitted changes, so this tag does"
  echo "             not identify a reproducible commit."
fi

echo "==> Enabling APIs"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com cloudscheduler.googleapis.com \
  secretmanager.googleapis.com storage.googleapis.com \
  --project "${PROJECT_ID}"

echo "==> Database URL in Secret Manager"
if ! gcloud secrets describe "${SECRET_NAME}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  : "${DATABASE_URL:?secret ${SECRET_NAME} does not exist yet -- set DATABASE_URL once to create it}"
  gcloud secrets create "${SECRET_NAME}" \
    --replication-policy automatic --project "${PROJECT_ID}" >/dev/null
  printf '%s' "${DATABASE_URL}" |
    gcloud secrets versions add "${SECRET_NAME}" \
      --data-file=- --project "${PROJECT_ID}" >/dev/null
  echo "    created ${SECRET_NAME}"
elif [[ -n "${DATABASE_URL:-}" ]]; then
  CURRENT="$(gcloud secrets versions access latest --secret "${SECRET_NAME}" \
    --project "${PROJECT_ID}" 2>/dev/null || true)"
  if [[ "${CURRENT}" != "${DATABASE_URL}" ]]; then
    printf '%s' "${DATABASE_URL}" |
      gcloud secrets versions add "${SECRET_NAME}" \
        --data-file=- --project "${PROJECT_ID}" >/dev/null
    echo "    added a new version of ${SECRET_NAME}"
  else
    echo "    unchanged"
  fi
else
  echo "    using the stored value"
fi

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format 'value(projectNumber)')"
RUNTIME_SA="${RUNTIME_SERVICE_ACCOUNT:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"
gcloud secrets add-iam-policy-binding "${SECRET_NAME}" \
  --member "serviceAccount:${RUNTIME_SA}" \
  --role roles/secretmanager.secretAccessor \
  --project "${PROJECT_ID}" >/dev/null

echo "==> Staging bucket"
if ! gcloud storage buckets describe "gs://${BUCKET}" \
  --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET}" \
    --project "${PROJECT_ID}" \
    --location "${REGION}" \
    --uniform-bucket-level-access
  echo "    created gs://${BUCKET}"
else
  echo "    gs://${BUCKET} already exists"
fi

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member "serviceAccount:${RUNTIME_SA}" \
  --role roles/storage.objectUser \
  --project "${PROJECT_ID}" >/dev/null

echo "==> Artifact Registry repository"
gcloud artifacts repositories describe "${REPOSITORY}" \
  --location "${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1 ||
  gcloud artifacts repositories create "${REPOSITORY}" \
    --repository-format docker \
    --location "${REGION}" \
    --project "${PROJECT_ID}" \
    --description "urban-ml container images"

echo "==> Cleanup policy"
CLEANUP_POLICY="$(mktemp)"
trap 'rm -f "${CLEANUP_POLICY}"' EXIT
cat >"${CLEANUP_POLICY}" <<'JSON'
[
  {
    "name": "delete-untagged",
    "action": {"type": "Delete"},
    "condition": {"tagState": "untagged", "olderThan": "7d"}
  },
  {
    "name": "keep-recent",
    "action": {"type": "Keep"},
    "mostRecentVersions": {"keepCount": 2}
  }
]
JSON
gcloud artifacts repositories set-cleanup-policies "${REPOSITORY}" \
  --location "${REGION}" --project "${PROJECT_ID}" \
  --policy "${CLEANUP_POLICY}" --no-dry-run >/dev/null

echo "==> Building image"
gcloud builds submit --tag "${IMAGE}" --project "${PROJECT_ID}" .

deploy_api() {
  echo "==> Deploying ${API_SERVICE}"
  gcloud run deploy "${API_SERVICE}" \
    --image "${IMAGE}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    "${API_AUTH_FLAG}" \
    --command uvicorn \
    --args "urban_ml.api.main:app,--host,0.0.0.0,--port,8080" \
    --port 8080 \
    --memory 1Gi \
    --cpu 1 \
    --cpu-boost \
    --min-instances 0 \
    --max-instances 2 \
    --concurrency 40 \
    --timeout 60 \
    --startup-probe "httpGet.path=/health,initialDelaySeconds=0,periodSeconds=5,timeoutSeconds=5,failureThreshold=12" \
    --clear-secrets \
    --set-env-vars "SYSTEM_ID=${SYSTEM_ID},MODEL_DIR=/app/models/current,APP_ENV=production,GCS_BUCKET=${BUCKET}"
}

deploy_ingest() {
  echo "==> Deploying ${INGEST_SERVICE} (private: only the scheduler may invoke it)"
  gcloud run deploy "${INGEST_SERVICE}" \
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
    --set-secrets "DATABASE_URL=${SECRET_NAME}:latest" \
    --set-env-vars "SYSTEM_ID=${SYSTEM_ID},APP_ENV=production,GCS_BUCKET=${BUCKET}"
}

ensure_scheduler() {
  local service_url
  service_url="$(gcloud run services describe "${INGEST_SERVICE}" \
    --region "${REGION}" --project "${PROJECT_ID}" --format 'value(status.url)')"

  echo "==> Service account for the scheduler"
  gcloud iam service-accounts describe \
    "${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --project "${PROJECT_ID}" >/dev/null 2>&1 ||
    gcloud iam service-accounts create "${SERVICE_ACCOUNT}" \
      --display-name "Invokes urban-ml ingestion" --project "${PROJECT_ID}"

  gcloud run services add-iam-policy-binding "${INGEST_SERVICE}" \
    --region "${REGION}" --project "${PROJECT_ID}" \
    --member "serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role roles/run.invoker >/dev/null

  echo "==> Scheduler job, every 5 minutes"
  local args=(
    --location "${REGION}"
    --project "${PROJECT_ID}"
    --schedule "*/5 * * * *"
    --uri "${service_url}/ingest"
    --http-method POST
    --oidc-service-account-email "${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"
    --oidc-token-audience "${service_url}"
    --attempt-deadline 120s
    --max-retry-attempts 1
  )
  if gcloud scheduler jobs describe "${SCHEDULER_JOB}" \
    --location "${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud scheduler jobs update http "${SCHEDULER_JOB}" "${args[@]}"
  else
    gcloud scheduler jobs create http "${SCHEDULER_JOB}" "${args[@]}"
  fi
}

if [[ "${TARGET}" == "all" || "${TARGET}" == "ingest" ]]; then
  deploy_ingest
  ensure_scheduler
fi

if [[ "${TARGET}" == "all" || "${TARGET}" == "api" ]]; then
  deploy_api
  API_URL="$(gcloud run services describe "${API_SERVICE}" \
    --region "${REGION}" --project "${PROJECT_ID}" --format 'value(status.url)')"
fi

echo
echo "Done. Deployed ${IMAGE}"
if [[ -n "${API_URL:-}" ]]; then
  echo "  API at ${API_URL}"
  echo "  curl ${API_URL}/health"
  echo "  curl ${API_URL}/ready"
fi
echo "  gcloud run revisions list --region ${REGION} --project ${PROJECT_ID}"
echo "  roll back: gcloud run services update-traffic SERVICE --to-revisions REVISION=100 --region ${REGION}"
