from flask import Flask, request, jsonify
from fastai.basics import load_learner
from PIL import Image
from pathlib import Path
from fastai.vision.utils import PILImage
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

@app.route('/easter-egg')
def egg():
    return 'You found an easter egg!🚀'

# load learner/model
path = Path('./dl_model/')
learn = load_learner(fname=path/'export.pkl')


def predict_single(im):
    # predicts single inputed user image
    pred,pred_idx,probs = learn.predict(im)
    result = {
        'label': pred,
        'probability': f'{probs[pred_idx]:.04f}',
    }
    return json.dumps(result)


import torchvision.transforms as tfms

@app.route('/', methods=['POST'])
def predict():
    img = request.files['image']
    im = PILImage.create(img)
    return predict_single(im)

if __name__ == '__main__':
    app.run()
