#!/usr/bin/env bash
# One-time GCP project setup for the IT Support RAG pipeline.
# Run this once after creating your GCP project.
# Usage: bash gcp_setup.sh <PROJECT_ID> [REGION]

set -euo pipefail

PROJECT_ID="${1:?Usage: bash gcp_setup.sh <PROJECT_ID> [REGION]}"
REGION="${2:-us-central1}"
REPO="it-support-rag"
SERVICE="it-support-rag"

echo "==> Setting project: ${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}"

echo "==> Enabling required APIs..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com

echo "==> Creating Artifact Registry repository..."
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="IT Support RAG Docker images" \
  2>/dev/null || echo "Repository already exists, skipping."

echo "==> Creating secrets in Secret Manager..."
echo "Enter your GROQ_API_KEY:"
read -rs GROQ_KEY
printf '%s' "${GROQ_KEY}" | gcloud secrets create groq-api-key \
  --data-file=- --replication-policy=automatic 2>/dev/null \
  || printf '%s' "${GROQ_KEY}" | gcloud secrets versions add groq-api-key --data-file=-

echo "Enter your DEMO_API_KEY:"
read -rs DEMO_KEY
printf '%s' "${DEMO_KEY}" | gcloud secrets create demo-api-key \
  --data-file=- --replication-policy=automatic 2>/dev/null \
  || printf '%s' "${DEMO_KEY}" | gcloud secrets versions add demo-api-key --data-file=-

echo "==> Granting Cloud Build access to secrets and Cloud Run..."
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')
CB_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/iam.serviceAccountUser"

echo "Setup complete"

