## Training setup

### Create the Python virtual environment

```bash
python3 -m venv custom-llm-env
source custom-llm-env/bin/activate
pip install -r requirements.txt
```

### Run a sample training session

```bash
python data.py --make_sample
python train.py --steps 200
```

## Sync corpus to S3

```bash
chmod +x sync_corpus_to_s3.sh
./sync_corpus_to_s3.sh
```

## Sync the latest checkpoint from S3

```bash
chmod +x sync_checkpoints.sh
./sync_checkpoints.sh
```

### What it does

1. Deletes everything under: `s3://bitnet-training-456088019014-us-east-1-an/corpus/`
2. Uploads all files from: `./corpus`
3. Lists the uploaded objects for confirmation.

### Notes

- `aws s3 sync` preserves folder structure.
- Large files automatically use multipart uploads.
- Your SageMaker job will then pull this dataset into: `/opt/ml/input/data/train` when the training job starts.
