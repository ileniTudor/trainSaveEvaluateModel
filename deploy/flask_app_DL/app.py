import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template, send_from_directory, jsonify
from torch import nn
from torchvision.models import resnet18
from werkzeug.utils import secure_filename
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = resnet18(pretrained=False, num_classes=2)
model.to(DEVICE)

state_dict = torch.load('best_model_resnet18.pth',  map_location=torch.device(device=DEVICE))
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Assuming 2 classes: cat and dog

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            label = predict(filepath)
            return jsonify({'label': label})
    return render_template('index.html')

def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    _, predicted = output.max(1)
    # Assuming 0 is cat and 1 is dog in the model's output classes
    return 'Dog' if predicted.item() == 1 else 'Cat'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)
