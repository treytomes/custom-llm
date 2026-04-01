#!/usr/bin/env bash
set -e

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BUCKET="bitnet-training-456088019014-us-east-1-an"
PREFIX="corpus"
LOCAL_DIR="./corpus"

S3_URI="s3://${BUCKET}/${PREFIX}"

# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------
if [ ! -d "${LOCAL_DIR}" ]; then
    echo "ERROR: Local corpus directory not found: ${LOCAL_DIR}"
    exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
    echo "ERROR: AWS CLI is not installed."
    exit 1
fi

echo ""
echo "Local corpus directory: ${LOCAL_DIR}"
echo "Target S3 location: ${S3_URI}"
echo ""

# ------------------------------------------------------------
# Step 1 — Delete existing S3 corpus
# ------------------------------------------------------------
echo "Deleting existing S3 corpus..."
aws s3 rm "${S3_URI}" --recursive || true

# ------------------------------------------------------------
# Step 2 — Upload local corpus
# ------------------------------------------------------------
echo "Uploading local corpus..."
aws s3 sync "${LOCAL_DIR}" "${S3_URI}"

echo ""
echo "Corpus synchronization complete."
echo ""

# ------------------------------------------------------------
# Show result
# ------------------------------------------------------------
aws s3 ls "${S3_URI}" --recursive
