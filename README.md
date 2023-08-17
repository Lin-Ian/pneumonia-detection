# pneumonia-detection

Applying transfer learning to detect pneumonia with a chest x-ray image.

## Technologies
This project is created with:
- Python 3.11
- TensorFlow
- VGG16 model
- Google Colab

## Challenges
The original Kaggle dataset have a very small validation dataset size.
To improve the neural network performance, I had to increase the validation dataset size.
I split the data between the validation and testing dataset into an even 50/50 split.
This improved the training validation performance.

## Outcomes
After training the neural network, I was able to achieve 91.6% accuracy, 88.4% precision, and 99.5% recall.

## Acknowledgements
- [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [TensorFlow Course – Building and Evaluating Medical AI Models](https://www.youtube.com/watch?v=8m3LvPg8EuI)