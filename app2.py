from flask import Flask, request, jsonify
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
from flask_cors import CORS  # Import CORS

from ann2 import NeuralNetwork  # Import the model definition

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the model architecture
input_size = 400  # Based on your model's input size (20x20 flattened)
model = NeuralNetwork(input_size)
model.load_state_dict(torch.load('model2.pth'))
model.eval()  # Set the model to evaluation mode

# Define the transformation for image preprocessing
transform = transforms.Compose([
    transforms.Resize((20, 20)),  # Resize to the size expected by the model
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale (1 channel)
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize image
])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the file from the POST request
        file = request.files['file']
        
        # Open the image file using PIL
        image = Image.open(file)

        # Apply the transformations to the image
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.view(-1, 400) 

        # Perform inference (prediction) without tracking gradients
        with torch.no_grad():
            output = model(image)

        # Apply sigmoid since this is a binary classification problem
        output = torch.sigmoid(output)

        # Get the prediction label and probability
        pothole_prob = output.item()  # Since it's a single output, get the scalar value
        normal_prob = 1 - pothole_prob  # For binary classification, the second class is the complement

        # Determine the prediction label based on the probabilities
        prediction = "Pothole" if pothole_prob > normal_prob else "Normal"

        # Return the prediction result as JSON
        return jsonify({
            "prediction": prediction,
            "class_probabilities": {
                "Pothole": pothole_prob,
                "Normal": normal_prob
            }
        })

    except Exception as e:
        # Handle any exceptions and return an error message
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Change to any available port, e.g., 5000

