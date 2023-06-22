import argparse
from keras import models
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Define the command line arguments
parser = argparse.ArgumentParser(description='Colon Cancer Prediction')
parser.add_argument('image_path', metavar='path', type=str, help='Path to the image file')
parser.add_argument('model_path', metavar='path', type=str, help='Path to the trained model file')
args = parser.parse_args()

# Load the trained model
model = models.load_model(args.model_path)

# Load and preprocess the input image
input_image = Image.open(args.image_path).convert('RGB')
resized_image = input_image.resize((224, 224))
preprocessed_image = np.array(resized_image)
preprocessed_image = preprocess_input(preprocessed_image)
preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

# Predict the class probabilities
predictions = model.predict(preprocessed_image)[0]
predicted_class_index = np.argmax(predictions)
predicted_class_label = ['Cancerous', 'Non-Cancerous'][predicted_class_index]
score = predictions[predicted_class_index]

# Display the input image and the predicted class label with score
plt.imshow(Image.open(args.image_path))
plt.title(f'Predicted class: {predicted_class_label} with score: {score}')
plt.axis('on')
plt.savefig('prediction.png')
plt.show()
