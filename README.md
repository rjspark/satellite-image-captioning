# 🛰️ AI-Powered Satellite Image Captioning

<div align="center">

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Hugging_Face-yellow)](https://rjspark-satellite-image-captioning.hf.space)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Advanced Deep Learning System for Automated Natural Language Description Generation from Satellite Imagery**

[Live Demo](https://rjspark-satellite-image-captioning.hf.space) • [Report Bug](https://github.com/rjspark/satellite-image-captioning/issues) • [Request Feature](https://github.com/rjspark/satellite-image-captioning/issues)

</div>

---

## 📋 Overview

This project implements a production-ready deep learning framework for automated caption generation from satellite imagery. By integrating **Convolutional Neural Networks (CNNs)** for visual feature extraction and **Long Short-Term Memory (LSTM)** networks for sequence generation, the system interprets complex remote sensing data to produce context-aware natural language descriptions.

### ✨ Key Highlights

- 🧠 **ResNet50 + LSTM Architecture** - State-of-the-art encoder-decoder model
- 🚀 **Real-time Processing** - Fast inference for operational applications
- 🎨 **Professional UI** - Modern, responsive web interface
- 🌍 **Real-world Applications** - Urban planning, environmental monitoring, disaster response
- 🐳 **Docker Deployed** - Production-ready containerized application

---

## 🎯 Demo

**Try it live:** [https://rjspark-satellite-image-captioning.hf.space](https://rjspark-satellite-image-captioning.hf.space)

### Sample Results

Upload a satellite image → Get AI-generated description in seconds!

```
Input: Satellite image of urban area
Output: "Dense urban region with residential buildings and road networks visible"
```

---

## 🏗️ Architecture

### Model Pipeline

```
Satellite Image (224×224)
        ↓
   CNN Encoder (ResNet50)
        ↓
   Feature Vector (256-dim)
        ↓
   LSTM Decoder (512 units)
        ↓
   Natural Language Caption
```

### Technical Specifications

| Component | Technology | Details |
|-----------|------------|---------|
| **Encoder** | ResNet50 | Pre-trained CNN for feature extraction |
| **Decoder** | LSTM | 512 hidden units, 1 layer |
| **Embedding** | 256 dimensions | Word embeddings |
| **Vocabulary** | ~1,100 words | Domain-specific vocabulary |
| **Framework** | PyTorch 2.1.0 | Deep learning framework |
| **Backend** | Flask 3.0.0 | Web application server |
| **Frontend** | HTML/CSS/JS | Responsive modern UI |
| **Deployment** | Docker | Containerized on Hugging Face Spaces |

---

## 🚀 Features

### 🤖 AI Capabilities
- **Visual Feature Extraction**: Deep CNN captures spatial patterns and semantic information
- **Natural Language Generation**: LSTM generates coherent, context-aware descriptions
- **Transfer Learning**: Leverages pre-trained ResNet50 for robust feature extraction
- **Sequence Modeling**: Captures long-term dependencies in caption generation

### 💻 Web Application
- **Drag-and-drop Upload**: Intuitive image upload interface
- **Real-time Processing**: Fast caption generation (2-3 seconds)
- **Professional Design**: Dark-themed, modern UI with animations
- **Responsive Layout**: Works perfectly on desktop and mobile
- **Error Handling**: Graceful error messages and validation

### 🌍 Real-World Applications

#### 🏙️ Urban Planning
- Infrastructure mapping and analysis
- City development monitoring
- Transportation network assessment

#### 🌿 Environmental Monitoring
- Deforestation tracking
- Land use change detection
- Ecosystem health assessment

#### ⚠️ Disaster Response
- Post-disaster damage assessment
- Infrastructure integrity evaluation
- Emergency response support

#### 🌾 Agriculture & Forestry
- Crop classification
- Yield prediction
- Forest management

---

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.9+** - Programming language
- **PyTorch 2.1.0** - Deep learning framework
- **torchvision 0.16.0** - Computer vision utilities
- **Flask 3.0.0** - Web framework
- **NumPy 1.26.4** - Numerical computing

### Development Tools
- **Git** - Version control
- **Docker** - Containerization
- **Git LFS** - Large file storage

### Deployment
- **Hugging Face Spaces** - Cloud hosting
- **Docker Container** - Production environment

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git (for cloning)

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/rjspark/satellite-image-captioning.git
cd satellite-image-captioning
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Open in browser**
```
http://localhost:7860
```

---

## 🎮 Usage

### Web Interface

1. Navigate to the [live demo](https://rjspark-satellite-image-captioning.hf.space)
2. Click the upload area or drag and drop a satellite image
3. Wait for the image preview to load
4. Click "Generate Caption with AI"
5. View the AI-generated description

### Supported Image Formats
- JPG / JPEG
- PNG
- Max file size: 16MB
- Recommended: 224×224 pixels or larger

### API Usage (Local)

```python
from PIL import Image
from app import generate_caption

# Load image
image_path = "path/to/satellite/image.jpg"

# Generate caption
caption = generate_caption(image_path)
print(f"Caption: {caption}")
```

---

## 🧪 Model Training

### Dataset
The model was trained on a custom dataset of satellite images with corresponding captions.

### Training Details
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss
- **Batch Size**: 32
- **Epochs**: Multiple iterations until convergence
- **Hardware**: GPU-accelerated training

### Model Files
- `encoder.pth` - Trained ResNet50 encoder weights (96 MB)
- `decoder.pth` - Trained LSTM decoder weights (10 MB)
- `vocab.pkl` - Vocabulary object (~1,100 words)

---

## 📊 Project Structure

```
satellite-image-captioning/
│
├── app.py                  # Flask application (main)
├── model.py                # Encoder-Decoder architecture
├── vocabulary.py           # Vocabulary class definition
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
│
├── encoder.pth             # Trained encoder weights
├── decoder.pth             # Trained decoder weights
├── vocab.pkl               # Vocabulary object
│
├── templates/
│   └── index.html          # Web interface (frontend)
│
└── uploads/                # Temporary upload directory
```

---

## 🔬 Technical Details

### Encoder (ResNet50)
- Pre-trained on ImageNet
- Final FC layer removed for feature extraction
- Outputs 2048-dimensional feature vectors
- Batch normalization for stable features

### Decoder (LSTM)
- 512 hidden units
- Single layer architecture
- Embedding dimension: 256
- Max sequence length: 20 words
- Greedy decoding for inference

### Image Preprocessing
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## 🚀 Deployment

### Hugging Face Spaces

The application is deployed on Hugging Face Spaces using Docker:

1. **Build Docker Image**: Automated on push to main branch
2. **Install Dependencies**: Requirements installed in container
3. **Load Models**: Model weights loaded at startup
4. **Run Application**: Flask server starts on port 7860

### Docker Configuration

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- Model improvements (attention mechanism, transformer architecture)
- UI/UX enhancements
- Additional features (batch processing, multi-language support)
- Documentation improvements
- Bug fixes

---

## 📈 Future Enhancements

- [ ] **Attention Mechanism** - Improve caption quality with visual attention
- [ ] **Beam Search** - Generate multiple caption candidates
- [ ] **Transformer Architecture** - Upgrade to Vision Transformer + GPT
- [ ] **Multi-language Support** - Captions in multiple languages
- [ ] **Batch Processing** - Process multiple images simultaneously
- [ ] **Confidence Scores** - Display model confidence for predictions
- [ ] **REST API** - Programmatic access for developers
- [ ] **Mobile App** - Native iOS and Android applications
- [ ] **Fine-tuning Interface** - Allow users to fine-tune on custom datasets

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**RJ Spark**

- GitHub: [@rjspark](https://github.com/rjspark)
- Hugging Face: [@rjspark](https://huggingface.co/rjspark)
- LinkedIn: [Add your LinkedIn]
- Email: [Add your email]

---

## 🙏 Acknowledgments

- **PyTorch Team** - Deep learning framework
- **ResNet Architecture** - He et al. (2015)
- **LSTM Networks** - Hochreiter & Schmidhuber (1997)
- **Hugging Face** - Deployment platform
- **Remote Sensing Community** - Dataset and domain knowledge


## 📞 Support

If you encounter any issues or have questions:

- 🐛 [Open an Issue](https://github.com/rjspark/satellite-image-captioning/issues)
- 💬 [Start a Discussion](https://github.com/rjspark/satellite-image-captioning/discussions)
- 📧 Email: rjshreeya@gmail.com

---



**Built with ❤️ using PyTorch and Deep Learning**

[⬆ Back to Top](#-ai-powered-satellite-image-captioning)

</div>
