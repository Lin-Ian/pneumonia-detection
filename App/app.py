from flask import Flask, render_template, request
from keras.models import load_model
from model import predict_image
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
model = load_model('model.h5')


@app.route("/")
def home():
    return render_template('home.html')


@app.route("/upload", methods=['POST', 'GET'])
def upload():
    # Get uploaded image
    image = request.files['image']
    image_name = image.filename
    image.save(image_name)

    # Save image to pass to HTML template
    im = Image.open(image_name)
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())

    # Make prediction on image
    prediction, normal_confidence, pneumonia_confidence = predict_image(model, image_name)

    # Convert decimal to percentage
    normal_confidence = "{:.2%}".format(normal_confidence)
    pneumonia_confidence = "{:.2%}".format(pneumonia_confidence)

    # Remove image from root folder
    os.remove(image_name)

    return render_template('prediction.html',
                           img_data=encoded_img_data.decode('utf-8'), prediction=prediction,
                           normal_confidence=normal_confidence, pneumonia_confidence=pneumonia_confidence)


if __name__ == "__main__":
    app.run(debug=True)
