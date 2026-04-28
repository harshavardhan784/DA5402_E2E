#!/usr/bin/env bash
# Restores a past week's exact pipeline state from its git tag.
# cmd: ./reproduce.sh week2/run-abc12345

set -e

TAG=$1
if [ -z "$TAG" ]; then
  echo "Usage: $0 <git-tag>  e.g. week2/run-abc12345"
  exit 1
fi

echo "==> Checking out tag: $TAG"
git checkout "$TAG"

echo "==> Restoring data files for this commit"
dvc pull --force

# Extract original run_id from the tag annotation
RUN_ID=$(git tag -l --format='%(contents)' "$TAG" \
         | grep -oP '(?<=MLflow run_id=)\S+' || echo "unknown")

echo "==> Done. Week state fully restored."
echo "    params.yaml: $(cat params.yaml)"
echo "    MLflow run_id: $RUN_ID"    
echo "    Find run at: ${MLFLOW_TRACKING_URI:-http://localhost:5000}/#/experiments"