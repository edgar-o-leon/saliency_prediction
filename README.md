E-commerce human saliency project for Advizion.net. The model serves to predict human attention on e-commerce images. Based on the saliency model from the academic paper "TranSalNet: Towards perceptually relevant visual saliency prediction" < https://doi.org/10.1016/j.neucom.2022.04.080 > Deployed a TranSalNet model in ONNX format and trained on SALECI dataset. Deployment with AWS Sagemaker Serverless End Points, built on a Pytorch Inference Image (DLC), and an inference script with image processing functions.  It serves a response to an e-commerce image submitted by a user via the endpoint, which returns a saliency map image and provides an indication of the effectiveness of the e-commerce image. 

Endpoints allow for serving to a high number of concurrent requests. This deployment used serverless endpoints, in order to reduce costs.

The collab files included were created for proof of concept, local development of inference script functions, and final proof of working model by calling the endpoints via the boto3 API. 

Sagemaker Framework description:
The AWS Inferentia framework supports ONNX files. AWS Inferentia is designed to accelerate machine learning inference workloads and it supports popular frameworks like PyTorch, TensorFlow, and ONNX. This means you can deploy and run ONNX models on AWS Inferentia instances.

The model directory is compressed into a tar.gz file and loaded to AWS Sagemaker.

docker image selected: 
763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.4.0-cpu-py311

S3 bucket address
s3://sagemaker-us-west-2-054037098477/inference2-python-2024-11-12/model.tar.gz

Sagemaker Environmental variables: 
SAGEMAKER_PROGRAM: inference.py
SAGEMAKER_SUBMIT_DIRECTORY: /opt/ml/model/code
SAGEMAKER_CONTAINER_LOG_LEVEL: 20
SAGEMAKER_REGION: us-west-2
MMS_DEFAULT_RESPONSE_TIMEOUT:500