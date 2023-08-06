import os

os.environ["LOCALSTACK_ENDPOINT_URL"] = "http://134.209.75.14:4566"
os.environ["AWS_ACCESS_KEY_ID"] = "foo"
os.environ["AWS_SECRET_ACCESS_KEY"] = "bar"
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
