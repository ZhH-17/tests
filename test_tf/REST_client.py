from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import requests
import json
import os
import time
import sys

import pdb
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

host = "0.0.0.0"
port = "8501"
model_name = ""
model_names = ["test_model", "hr_strip",]
url = 'http://{0}:{1}/v1/models/{2}:predict'. \
    format(host, port, model_names[0])
test = np.loadtxt("./iris_test.csv", skiprows=1, delimiter=",")
test_x, test_y = test[:, :-1], test[:, -1]

input_key = "input_1:0"
output_key = "dense2:BiasAdd:0"
instance = {input_key: test_x[0]}
instances = [{input_key: x} for x in test_x]
# param = {'signature_name': 'detection', "instances": test_x}
param = {'signature_name': 'detection', "instances": instances}
param = json.dumps(param, cls=NumpyEncoder)

res = requests.post(url, data=param)
if 'error' in res.json():
    print(res.json())
    sys.exit()
outputs = res.json()['predictions']
print("input {} samples".format(len(test_x)))
print("output {} results".format(len(outputs)))
print(np.array(outputs))
