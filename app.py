from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import pickle
import os
from model import Encoder, Decoder
from vocabulary import Vocabulary

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model parameters (match your training configuration)
embed_size = 256
hidden_size = 512
num_layers = 1

print("Loading model components...")

# Load vocabulary
try:
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    print(f"✓ Vocabulary loaded successfully ({len(vocab)} words)")
except Exception as e:
    print(f"✗ Error loading vocabulary: {e}")
    raise

vocab_size = len(vocab)

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Using device: {device}")

# Initialize models
try:
    encoder = Encoder(embed_size).to(device)
    decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers).to(device)
    print("✓ Model architecture initialized")
except Exception as e:
    print(f"✗ Error initializing models: {e}")
    raise

# Load trained weights
try:
    encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
    decoder.load_state_dict(torch.load('decoder.pth', map_location=device))
    print("✓ Model weights loaded successfully")
except Exception as e:
    print(f"✗ Error loading model weights: {e}")
    raise

# Set models to evaluation mode
encoder.eval()
decoder.eval()
print("✓ Models ready for inference")

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def generate_caption(image_path):
    """
    Generate natural language caption for satellite image
    
    Args:
        image_path: Path to the satellite image file
        
    Returns:
        Generated caption as string
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate caption using encoder-decoder architecture
        with torch.no_grad():
            # Extract visual features using CNN encoder
            features = encoder(image_tensor)
            
            # Generate word sequence using LSTM decoder
            sampled_ids = decoder.sample(features)
            sampled_ids = sampled_ids[0].cpu().numpy()
        
        # Convert word IDs to natural language
        caption = []
        for word_id in sampled_ids:
            word = vocab.itos[word_id]
            if word == "<EOS>":  # End of sequence
                break
            if word not in ["<PAD>", "<SOS>"]:  # Skip special tokens
                caption.append(word)
        
        return " ".join(caption).capitalize()
    
    except Exception as e:
        raise Exception(f"Caption generation failed: {str(e)}")

@app.route('/')
def home():
    """Render the main application interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for satellite image caption generation
    
    Accepts: multipart/form-data with 'file' field
    Returns: JSON with generated caption or error message
    """
    # Validate file upload
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file type
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return jsonify({'error': 'Invalid file type. Please upload JPG, PNG, or JPEG'}), 400
    
    # Process the uploaded image
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        
        try:
            # Save uploaded file
            file.save(filepath)
            print(f"Processing image: {file.filename}")
            
            # Generate caption using deep learning model
            caption = generate_caption(filepath)
            print(f"Generated caption: {caption}")
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'caption': caption,
                'success': True
            })
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            
            print(f"Error processing image: {str(e)}")
            return jsonify({
                'error': f'Processing error: {str(e)}',
                'success': False
            }), 500

@app.route('/health')
def health():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'model': 'ResNet50-LSTM',
        'device': str(device),
        'vocab_size': vocab_size
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🛰️  SATELLITE IMAGE CAPTIONING AI")
    print("="*60)
    print(f"Server starting on http://0.0.0.0:7860")
    print(f"Model: ResNet50 Encoder + LSTM Decoder")
    print(f"Vocabulary size: {vocab_size} words")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=7860)
