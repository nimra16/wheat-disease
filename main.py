from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from flask_restful import Resource, Api, reqparse
import os
from tensorflow.keras.preprocessing import image
import cv2
app = Flask(__name__)
model = load_model('EfficientNetB2-80-train-aug-wheat-97.56.h5')
target_img = os.path.join(os.getcwd(), 'static/images')


@app.route('/')
def index_view():
    return render_template('index.html')


# Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'JPG', 'PNG'])


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXT


# Function to load and prepare the image in right shape
def read_image(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (300,300))
    # load_img(filename, target_size=(300, 300,3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

labels = ["Healthy", "Septoria", "Stripe Rust"]
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):  # Checking file format
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path)  # prepressing method
            # print(img)
            class_prediction = model.predict(img)
            classes_x = np.argmax(class_prediction, axis=1)

            fruit = str(labels[classes_x[0]])
            # 'fruit' , 'prob' . 'user_image' these names we have seen in predict.html.
            return render_template('predict.html', fruit=fruit, prob=class_prediction, user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8001)
