from typing_extensions import NamedTuple
import io
from io import BytesIO
from PIL import Image
from flask import Flask,request, jsonify, Response
import base64
from ultralytics import YOLO
import json
import re

app = Flask(__name__)
model = YOLO('best.pt')

cache = {}

@app.route("/", methods=["GET"])
def send():
    return jsonify({"name":cache['name']})

@app.route("/objectdetection", methods=["POST"])
def predict():
    # if request.method == "POST":
        # img = request.form["image"]
    cache['name'] = 'none'
    
    json_data = request.get_json()
    image_data = re.sub('^data:image/.+;base64,', '', json_data)
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    # dict_data = json.loads(json_data)
    # data_as_dictionary = json.loads(json.loads(json_data.decode('utf-8')))
    # base64_img = data_as_dictionary['image']
    # bytes_img = encode(base64_img, 'utf-8')
    # binary_img = base64.decodebytes(bytes_img)
    # img = json_data['image']
    # img = base64.b64decode(img)
    # img = BytesIO(img)
    # img = Image.open(img)

    # image_file = request.files["image"]
    # image_bytes = image_file.read()
    # img = Image.open(io.BytesIO(image_bytes))
    results = model(img)
    
    names = model.names
    for r in results:
        for c in r.boxes.cls:
            print(names[int(c)])
            if names[int(c)] == 'cups' or names[int(c)] == 'tumbler':
                cache['name'] = names[int(c)]
                print(cache['name'])
                break
    
    return jsonify({"name":cache['name']})

if __name__ == '__main__':
    app.run('0.0.0.0',port=8000,debug=True)