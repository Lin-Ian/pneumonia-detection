from keras.models import load_model
import numpy as np
from PIL import Image
import os


def load_image_into_numpy_array(image):
    # Convert image to numpy array
    image = image.convert('RGB')
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def predict_image(model, filename):
    # Load and process image
    image = Image.open(filename).resize((224, 224))
    image_np = load_image_into_numpy_array(image)
    exp = np.true_divide(image_np, 255.0)
    expanded = np.expand_dims(exp, axis=0)

    # Predict confidence of normal or pneumonia diagnosis
    confidence = model.predict(expanded)[0]
    normal_confidence = confidence[0]
    pneumonia_confidence = confidence[1]

    # Output prediction
    if normal_confidence > pneumonia_confidence:
        return "normal", normal_confidence, pneumonia_confidence
    elif normal_confidence <= pneumonia_confidence:
        return "pneumonia", normal_confidence, pneumonia_confidence


def main():
    # Load model
    model = load_model('model.h5')

    # Image classes in images directory
    image_classes = os.listdir('images')

    # Go through each class in images directory
    for image_class in image_classes:

        # Create filepath to image class directory
        image_class_path = os.path.join('images', image_class)

        # Go through each image in the image class directory
        for image in os.listdir(image_class_path):

            # Create filepath to image
            filepath = os.path.join(image_class_path, image)
            # Make prediction
            prediction, normal_score, pneumonia_score = predict_image(model, filepath)

            # Output prediction
            print(f'Label: {image_class} Prediction: {prediction} '
                  f'Normal Confidence: {normal_score} Pneumonia Confidence: {pneumonia_score}\n')


if __name__ == '__main__':
    main()
