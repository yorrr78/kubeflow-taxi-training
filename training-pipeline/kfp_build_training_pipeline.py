
import json
import yaml

import kfp
from kfp.v2 import dsl
from kfp.v2 import compiler
from kfp.components import load_component_from_file


download_taxi_data = load_component_from_file("./components/ingestion-component.yaml")
preprocess_taxi_data = load_component_from_file("./components/preprocessing-component.yaml")
train_taxi_data = load_component_from_file("./components/training-component.yaml")
evaluate_model = load_component_from_file("./components/evaluating-component.yaml")
register_model = load_component_from_file("./components/registering-component.yaml")
deploy_model = load_component_from_file("./components/deploying-component.yaml")
test_prediction = load_component_from_file("./components/predicting-component.yaml")


#read configuration from file
with open("config.json") as json_file:
    config = json.load(json_file)

PIPELINE_NAME = config.get("pipeline_name")
PACKAGE_PATH = config.get("pipeline_package_path")
BUCKET_NAME = config.get("bucket_name") 
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root/"
    
@dsl.pipeline(
    name=PIPELINE_NAME,
    pipeline_root=PIPELINE_ROOT
)
def model_pipeline(
    project_id:str="",
    region:str="asia-northeast3",
    pipeline_name:str="",
    pipeline_package_path:str="",
    bucket_name:str="",
    train_size:float=0.8,
    valid_size:float=0.1,
    test_size:float=0.1,
    deployment_metric:str="r2",
    deployment_metric_threshold:float=0.7,
    serving_container_uri:str='asia-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest',
    model_name:str='sklearn-kubeflow-nytaxi-regression-model',
    params:dict={
        'max_depth': 20,
        'n_estimators': 100,
        'min_samples_leaf': 10,
        'random_state': 0
    },
    prediction_blob:str='predictions/predictions.json',
):
    ingestion_task = download_taxi_data(bucket_name)
    processing_task = preprocess_taxi_data(
        input_data=ingestion_task.output,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size
    )
    training_task = train_taxi_data(
        train_data=processing_task.outputs["train_data"],
        params=params,
    )
    evaluating_task = evaluate_model(
        val_data=processing_task.outputs["valid_data"],
        model=training_task.outputs["model"],
        target_column_name='duration',
        deployment_metric=deployment_metric,
        deployment_metric_threshold=deployment_metric_threshold,
    )
    
    with dsl.Condition(
        evaluating_task.outputs["dep_decision"] == "true",
        name="deploy_decision",
    ):
        # check the container uri list here: https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers#scikit-learn
        # deploy only if metric value exceeds deployment threshold
        registering_task = register_model(
            serving_container_uri=serving_container_uri,
            project_id=project_id,
            region=region,
            model_name=model_name,
            model=training_task.outputs["model"],
        )
        
        deploying_task = deploy_model(
            model_resource_name = registering_task.outputs["model_resource_name"],
            project_id=project_id,
            region=region
        )
        
        predicting_task = test_prediction(
            project_id=project_id,
            region=region,
            test_ds=processing_task.outputs["test_data"],
            endpoint_resource_name=deploying_task.outputs["endpoint_resource_name"],
            bucket_name=bucket_name,
            prediction_blob=prediction_blob,
        )

compiler.Compiler().compile(
    pipeline_func=model_pipeline, package_path=PACKAGE_PATH
)
