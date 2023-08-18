import numpy as np
from PIL import Image


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
