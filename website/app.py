from flask import Flask, jsonify, render_template
import numpy as np
import torch

app = Flask(__name__)

# Load your PyTorch LSTM model
model = torch.load('path_to_your_model.pth')
model.eval()  # Set the model to evaluation mode

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    # Generate or fetch real-time data
    data = np.random.rand(1, 10, 1)  # Example data, replace with real-time data
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(data_tensor)
    
    return jsonify(prediction.numpy().tolist())

if __name__ == '__main__':
    app.run(debug=True)