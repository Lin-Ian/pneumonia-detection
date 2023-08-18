from flask import Flask, render_template, request
from keras.models import load_model
from model import predict_image

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

    # Make prediction on image
    prediction, normal_confidence, pneumonia_confidence = predict_image(model, image_name)

    return f'{prediction} / {normal_confidence} / {pneumonia_confidence}'


if __name__ == "__main__":
    app.run(debug=True)
