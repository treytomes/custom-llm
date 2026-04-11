#!/usr/bin/env python3

import aws_cdk as cdk
import os
from pathlib import Path
from dotenv import load_dotenv

from cdk.scout_llm_stack import ScoutLlmStack
from util.config import Config


env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

config = Config()

app = cdk.App()
stack = ScoutLlmStack(app, "scout-llm-stack", config,
    # If you don't specify 'env', this stack will be environment-agnostic.
    # Account/Region-dependent features and context lookups will not work,
    # but a single synthesized template can be deployed anywhere.

    # Uncomment the next line to specialize this stack for the AWS Account
    # and Region that are implied by the current CLI configuration.

    #env=cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region=os.getenv('CDK_DEFAULT_REGION')),

    # Uncomment the next line if you know exactly what Account and Region you
    # want to deploy the stack to. */

    #env=cdk.Environment(account='123456789012', region='us-east-1'),

    # For more information, see https://docs.aws.amazon.com/cdk/latest/guide/environments.html
)

cdk.Tags.of(stack).add("project", config.project)
cdk.Tags.of(stack).add("owner", config.owner)
cdk.Tags.of(stack).add("environment", config.environment)

app.synth()
