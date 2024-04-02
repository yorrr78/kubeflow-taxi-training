
from google.cloud import aiplatform
import yaml
import json
from datetime import datetime, timedelta

with open("config.json") as json_file:
    config = json.load(json_file)

DISPLAY_NAME = config.get("pipeline_name")
PACKAGE_PATH = config.get("pipeline_package_path")
BUCKET_NAME = config.get("bucket_name") 
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root/"

job = aiplatform.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path=PACKAGE_PATH,
    pipeline_root=PIPELINE_ROOT,
    parameter_values=config,
)

# For test purpose, execute pipeline every min
job_schedule = job.create_schedule(
  display_name=DISPLAY_NAME,
  cron="* * * * *",
  max_concurrent_run_count=5,
  max_run_count=30,
)
