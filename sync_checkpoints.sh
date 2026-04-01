#!/usr/bin/env bash
set -e

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BUCKET="bitnet-training-456088019014-us-east-1-an"
LOCAL_DIR="./checkpoints"
FILE="latest.pt"

S3_URI="s3://${BUCKET}/checkpoints/${FILE}"
LOCAL_PATH="${LOCAL_DIR}/${FILE}"

# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------
if ! command -v aws >/dev/null 2>&1; then
    echo "ERROR: AWS CLI is not installed."
    exit 1
fi

mkdir -p "${LOCAL_DIR}"

echo ""
echo "Downloading checkpoint:"
echo "  ${S3_URI}"
echo "to:"
echo "  ${LOCAL_PATH}"
echo ""

# ------------------------------------------------------------
# Download latest checkpoint
# ------------------------------------------------------------
aws s3 cp "${S3_URI}" "${LOCAL_PATH}"

echo ""
echo "Checkpoint download complete."
echo ""

ls -lh "${LOCAL_PATH}"
