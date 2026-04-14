import os


class Config:
    def __init__(self):
        self.project = os.getenv("PROJECT")
        self.owner = os.getenv("OWNER")
        self.environment = os.getenv("ENVIRONMENT")

        self.azure_ai_endpoint = os.getenv("AZURE_AI_ENDPOINT")
        self.azure_ai_key = os.getenv("AZURE_AI_KEY")
        self.azure_model_id = os.getenv("AZURE_MODEL_ID")
