import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib

# Load the saved model
model = load_model('skin_cancer_classifier.h5')

# Load the LabelEncoder (used during training)
label_encoder = joblib.load('label_encoder.pkl')

# Preprocessing function for images (same as during training)
def preprocess_image(image_path, target_size=(150, 150)):
    img = load_img(image_path, target_size=target_size)  # Load and resize the image
    img = img_to_array(img) / 255.0  # Convert image to array and normalize pixel values
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Path to the new image you want to classify
image_path = r"C:\Users\visha\Downloads\archive\HAM10000_images_part_1\ISIC_0029302.jpg"  # Use raw string literal (r"")

# Preprocess the image
processed_image = preprocess_image(image_path)

# Predict the class probabilities for the image
predictions = model.predict(processed_image)

# Get the predicted class label (index of the highest probability)
predicted_class_index = np.argmax(predictions, axis=1)

# Decode the predicted class index to the original class name
predicted_class_label = label_encoder.inverse_transform(predicted_class_index)

print(f'Predicted class label: {predicted_class_label[0]}')
