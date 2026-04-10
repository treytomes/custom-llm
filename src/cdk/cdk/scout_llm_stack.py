from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
    aws_s3 as s3,
    RemovalPolicy,
)
from constructs import Construct


class ScoutLlmStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.scout_bucket = s3.Bucket(
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

        CfnOutput(
            self,
            "scout-llm-data-bucket-arn",
            value=self.scout_bucket.bucket_arn,
            description="ARN of the Scout LLM data backup bucket.",
            export_name="scout-llm-data-bucket-arn",
        )

        CfnOutput(
            self,
            "scout-llm-data-bucket-name",
            value=self.scout_bucket.bucket_name,
            description="ARN of the Scout LLM data backup bucket.",
            export_name="scout-llm-data-bucket-name",
        )
