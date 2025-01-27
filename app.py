from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image
import io
import os

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Set maximum file size for uploads (e.g., 16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define CNN Model (Ensure this matches your saved model's architecture)
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 32x32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 16x16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 8x8
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 4x4
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 256 * 4 * 4)  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Load the Pretrained Model
model_path = os.path.join(os.getcwd(), "model.pth")  # Ensure the model path is correct
model = EnhancedCNN().to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
else:
    print("Error: Model file not found.")
    exit(1)

# Define Transform for Input Images
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])  # Adjust normalization if using color images
])

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    # Check if the file is too large
    if file.mimetype.split('/')[0] != 'image':
        return jsonify({"error": "Uploaded file is not a valid image."}), 400

    try:
        # Read the uploaded file as an image
        image = Image.open(io.BytesIO(file.read())).convert('L').resize((64, 64))
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Predict using the model
        output = model(image_tensor)
        confidence = torch.sigmoid(output).item()
        prediction = "Pothole" if confidence > 0.5 else "Normal"

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "class_probabilities": {
                "Pothole": round(confidence, 2),
                "Normal": round(1 - confidence, 2)
            }
        }), 200

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")  # Log the error on the server
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Run the Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
