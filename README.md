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