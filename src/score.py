import json
import time
import sys
import os
from azureml.core.model import Model
import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime

def init():
    global session
    model = Model.get_model_path(model_name = 'tinyyolov2')
    session = onnxruntime.InferenceSession(model)

def preprocess(input_data_json):
    # convert the JSON data into the tensor input
    input_json = json.loads(input_data_json)
    input_array = np.array(input_json['data']).astype('float32')
    width = input_json["width"]
    height = input_json["height"]
    return np.reshape(input_array, (1, 3, width, height))

def postprocess(result):
    # convert numpy ndarray to list because numpy array is not JSON serializable
    return np.array(result[0]).tolist()

def run(input_data_json):
    try:
        start = time.time()   # start timer
        input_data = preprocess(input_data_json)
        input_name = session.get_inputs()[0].name  # get the id of the first input of the model   
        result = session.run([], {input_name: input_data})
        end = time.time()     # stop timer
        result_data = postprocess(result)
        return {"result": result_data,
                "time": end - start}
    except Exception as e:
        result = str(e)
        return {"error": result}