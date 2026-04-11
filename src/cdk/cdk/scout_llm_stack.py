import json
from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    SecretValue,
    Stack,
    aws_iam as iam,
    aws_s3 as s3,
    aws_secretsmanager as secretsmanager,
    aws_resourcegroups as resourcegroups
)
from constructs import Construct

from util.config import Config


class ScoutLlmStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, config: Config, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.data_bucket = s3.Bucket(
            self,
            "scout-llm-data",
            bucket_name="scout-llm-data",
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            lifecycle_rules=[
                s3.LifecycleRule(
                    enabled=True,
                    abort_incomplete_multipart_upload_after=Duration.days(7),
                    noncurrent_version_expiration=Duration.days(90)
                )
            ]
        )

        azure_secret_name = "scout-llm/azure"
        try:
            self.azure_secret = secretsmanager.Secret.from_secret_name_v2(
                self,
                "scout-llm-azure-secret",
                secret_name
            )
        except Exception:
            secret_payload = {
                "AZURE_AI_ENDPOINT": config.azure_ai_endpoint,
                "AZURE_AI_KEY": config.azure_ai_key,
                "AZURE_MODEL_ID": config.azure_model_id,
            }

            self.azure_secret = secretsmanager.Secret(
                self,
                "scout-llm-azure-secret",
                secret_name=azure_secret_name,
                secret_string_value=SecretValue.unsafe_plain_text(
                    json.dumps(secret_payload)
                )
            )
        
        self.sagemaker_role = iam.Role(
            self,
            "scout-llm-sagemaker-execution-role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Execution role for Scout LLM SageMaker training jobs."
        )
        self.sagemaker_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonSageMakerFullAccess"
            )
        )
        self.data_bucket.grant_read_write(self.sagemaker_role)
        self.azure_secret.grant_read(self.sagemaker_role)

        resourcegroups.CfnGroup(
            self,
            "scout-llm-resource-group",
            name="scout-llm-resources",
            resource_query={
                "Type": "TAG_FILTERS_1_0",
                "Query": json.dumps({
                    "ResourceTypeFilters": ["AWS::AllSupported"],
                    "TagFilters": [
                        {
                            "Key": "project",
                            "Values": [config.project]
                        }
                    ]
                })
            }
        )
        
        CfnOutput(
            self,
            "scout-llm-output-data-bucket-arn",
            value=self.data_bucket.bucket_arn,
            description="ARN of the Scout LLM data backup bucket.",
            export_name="data-bucket-arn",
        )

        CfnOutput(
            self,
            "scout-llm-output-data-bucket-name",
            value=self.data_bucket.bucket_name,
            description="ARN of the Scout LLM data backup bucket.",
            export_name="data-bucket-name",
        )

        CfnOutput(
            self,
            "scout-llm-output-azure-secret-arn",
            value=self.azure_secret.secret_arn,
            description="ARN of the secret Scout LLM uses to access Azure resources.",
            export_name="azure-secret-arn",
        )

        CfnOutput(
            self,
            "scout-llm-output-sagemaker-execution-role-arn",
            value=self.sagemaker_role.role_arn,
            description="IAM role used by SageMaker training jobs"
        )

