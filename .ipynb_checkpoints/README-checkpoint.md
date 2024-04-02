# Vertex AI Pipeline(Kubeflow) Training model pipeline

This sample kubeflow pipeline is for training the simple regression model.
</br>[Newyork taxi data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) used. </br>

This pipeline generate the model to predict the driving duration, 
based on pickup location (PULocationID), dropoff location(DOLocationID) and trip distance.

Generated model will be registered to Model Registry in Vertex AI,
This model will be deployed to Online Prediction in Vertex AI.

For scheduing test purpose, I used just random month from 2022 to see the whole process works fine
running every minute.

This Kubeflow training pipeline consist of steps following.


1. Download the data
2. Preprocessing
3. Training
4. Evaluating
5. Register the model
6. Deploy the model (Depending on the evaluation result)
7. Get test prediction results

![Flow Graph](./kubeflow-graph.png)



## Key files
```text
- training/kfp-training.ipynb: generating kfp components
- prediction_request.py: send reqeusts to deployed model

```