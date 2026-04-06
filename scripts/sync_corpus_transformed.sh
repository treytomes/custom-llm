aws s3 sync s3://bitnet-training-456088019014-us-east-1-an/corpus_transformed/ ./corpus_transformed/

# Push the chapters to S3.
# aws s3 sync ./chapters/ s3://bitnet-training-456088019014-us-east-1-an/chapters/

# Pull the chapters from S3.
# aws s3 sync s3://bitnet-training-456088019014-us-east-1-an/chapters/ ./chapters/
