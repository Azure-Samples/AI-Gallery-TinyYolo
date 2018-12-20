# ONNX YOLO Model javascript codes

## Background
TinyYOLO model takes input of numpy.ndarray and outputs numpy.ndarray. This directory contains helper javascript codes to do the conversion between image and arrays. Specifically, it does two things:

1. Take the input image file, manipulate the tensors and feed into the web service running ONNX TinyYOLO
2. Take the output of ONNX TinyYOLO, draw rectangles with prediction class names when the probabilities are high enough.

## How to Build
1. Go to the js directory (the current folder)
2. npm install 
3. npm run build
4. copy js/dist/bundle.js to ../static/, where the [01.deploy-onnx-yolo-model notebook](01.deploy-onnx-yolo-model.ipynb) will use to create docker image for inference

## About Azure ML
Get the full documentation for Azure Machine Learning service at https://docs.microsoft.com/azure/machine-learning/service/

## More Examples
 * [Azure/MachineLearningNotebooks GitHub site](https://github.com/Azure/MachineLearningNotebooks)
