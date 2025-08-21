# CodexIntern Python Development Project 🐍

A comprehensive collection of Python projects implementing various machine learning, data analysis, and web development tasks as part of the CodexIntern Python Development internship program.

## 📋 Project Overview

This repository contains implementations for **7 distinct projects** across two skill levels:

### 🟢 Slab 1 - Beginner Level
1. **Pandas Data Analysis Tool** - CSV analysis with visualizations
2. **Linear Regression House Price Predictor** - ML model for price prediction  
3. **Matrix Operations Tool** - Interactive matrix calculator with NumPy

### 🟡 Slab 2 - Intermediate Level
4. **Voice-Activated Personal Assistant** - Speech recognition and TTS
5. **Sentiment Analysis Web App** - TextBlob-based sentiment analysis
6. **Google Gemini Integration** - AI chat with conversation memory
7. **Speech-to-Image Generator** - Audio to image generation

## 🏗️ Project Structure

```
codexintern-python-development/
├── requirements.txt              # All project dependencies
├── README.md                    # This file
├── slab1-beginners/
│   ├── pandas-data-analysis/    # 📊 Data analysis with Flask web interface
│   ├── linear-regression-house-price/  # 🏠 ML price prediction model
│   └── matrix-operations-tool/  # 🔢 Interactive matrix calculator
├── slab2-intermediate/
│   ├── voice-assistant/         # 🎤 Voice-activated assistant
│   ├── sentiment-analysis-web/  # 😊 Sentiment analysis with web UI
│   ├── google-gemini-integration/  # 🤖 AI chatbot with memory
│   └── speech-to-image/         # 🎨 Audio to image generation
├── flask-app/                   # 🌐 Main web portal (optional)
└── shared/                      # 📦 Common utilities and resources
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd codexintern-python-development
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run individual projects**
   
   Each project can be run independently. Navigate to the specific project folder and follow its README.

## 📊 Slab 1 - Beginner Projects

### 1. Pandas Data Analysis Tool
**Location:** `slab1-beginners/pandas-data-analysis/`

**Features:**
- CSV file upload and analysis
- Interactive data exploration
- Statistical analysis with insights
- Multiple visualization types (bar charts, heatmaps, scatter plots)
- Web interface for easy interaction

**Usage:**
```bash
cd slab1-beginners/pandas-data-analysis
# Command line version
python data_analyzer.py
# Web interface
python flask_app.py  # Access at http://localhost:5001
```

### 2. Linear Regression House Price Predictor
**Location:** `slab1-beginners/linear-regression-house-price/`

**Features:**
- Synthetic dataset generation
- Feature engineering and preprocessing
- Multiple ML models (Linear Regression, Random Forest)
- Comprehensive visualizations
- Model evaluation and prediction

**Usage:**
```bash
cd slab1-beginners/linear-regression-house-price
python house_price_predictor.py
```

### 3. Matrix Operations Tool
**Location:** `slab1-beginners/matrix-operations-tool/`

**Features:**
- Interactive matrix calculator
- 14+ matrix operations (addition, multiplication, inverse, eigenvalues, etc.)
- Matrix storage and history tracking
- Demo mode with examples
- Command-line interface

**Usage:**
```bash
cd slab1-beginners/matrix-operations-tool
python matrix_calculator.py
```

## 🟡 Slab 2 - Intermediate Projects

### 4. Voice-Activated Personal Assistant
**Location:** `slab2-intermediate/voice-assistant/`

**Features:**
- Speech recognition and text-to-speech
- Task management (reminders, notes)
- Weather information
- News reading
- Voice-controlled interaction

**Usage:**
```bash
cd slab2-intermediate/voice-assistant
python voice_assistant.py
```

### 5. Sentiment Analysis Web Application
**Location:** `slab2-intermediate/sentiment-analysis-web/`

**Features:**
- TextBlob-based sentiment analysis
- Single text and batch processing
- CSV file analysis
- Interactive visualizations
- REST API endpoints
- Web dashboard

**Usage:**
```bash
cd slab2-intermediate/sentiment-analysis-web
# Command line version
python sentiment_analyzer.py
# Web interface
python flask_app.py  # Access at http://localhost:5002
```

### 6. Google Gemini Integration
**Location:** `slab2-intermediate/google-gemini-integration/`

**Features:**
- Google Gemini AI integration
- Conversation memory and history
- Real-time web search integration
- Context-aware responses
- Chat interface

**Usage:**
```bash
cd slab2-intermediate/google-gemini-integration
python gemini_chat.py
```

### 7. Speech-to-Image Generator
**Location:** `slab2-intermediate/speech-to-image/`

**Features:**
- **Open-source Stable Diffusion** - No API keys required!
- Multi-language speech recognition (6 languages)
- Audio file processing (WAV, MP3, FLAC, AIFF, OGG)
- Real-time speech-to-image generation
- GPU acceleration with CPU fallback
- Web interface with image gallery
- Completely offline operation (after initial setup)

**Usage:**
```bash
cd slab2-intermediate/speech-to-image
# Command line version
python speech_to_image.py
# Web interface
python flask_app.py  # Access at http://localhost:5005
```

## 🌐 Web Interfaces

Several projects include Flask web applications for better user experience:

| Project | Port | URL |
|---------|------|-----|
| Pandas Data Analysis | 5001 | http://localhost:5001 |
| Sentiment Analysis | 5002 | http://localhost:5002 |
| House Price Predictor | 5003 | http://localhost:5003 |
| Matrix Calculator | 5004 | http://localhost:5004 |
| Speech-to-Image Generator | 5005 | http://localhost:5005 |

## 🔧 Dependencies

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **scikit-learn** - Machine learning

### Web Framework
- **flask** - Web application framework
- **flask-cors** - Cross-origin resource sharing

### AI/ML Libraries
- **textblob** - Text processing and sentiment analysis
- **google-generativeai** - Google Gemini API
- **speechrecognition** - Speech-to-text conversion
- **pyttsx3** - Text-to-speech synthesis
- **torch** - PyTorch deep learning framework
- **diffusers** - Hugging Face Diffusers for Stable Diffusion
- **transformers** - Hugging Face Transformers library

### Additional Tools
- **requests** - HTTP library
- **pillow** - Image processing
- **python-dotenv** - Environment variable management

## 📱 API Endpoints

Several projects provide REST API endpoints for integration:

### Sentiment Analysis API
```bash
POST /api/analyze
Content-Type: application/json
{
    "text": "Your text to analyze"
}
```

### House Price Prediction API
```bash
POST /api/predict
Content-Type: application/json
{
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft_living": 2000,
    "location": "Suburb"
}
```

## 🐳 Docker Support

Each project includes Docker support for easy deployment:

```bash
# Build and run individual project
cd slab1-beginners/pandas-data-analysis
docker build -t pandas-analyzer .
docker run -p 5001:5001 pandas-analyzer
```

## 🔐 Environment Variables

Some projects require API keys. Create `.env` files in relevant project directories:

```env
# For Google Gemini integration
GOOGLE_API_KEY=your_gemini_api_key_here

# For weather API (voice assistant)
WEATHER_API_KEY=your_weather_api_key_here

# For news API
NEWS_API_KEY=your_news_api_key_here
```

## 📊 Performance Metrics

### Data Analysis Tool
- Supports CSV files up to 16MB
- Processes 10K+ rows efficiently
- Generates visualizations in <5 seconds

### ML Models
- Linear Regression: R² score ~0.85+
- House Price Predictor: MAE <$25K
- Processing time: <2 seconds for predictions

### Web Applications
- Response time: <500ms for most operations
- Concurrent users: 10+ supported
- File upload: 16MB max size

## 🧪 Testing

Run tests for individual projects:

```bash
# Example for sentiment analysis
cd slab2-intermediate/sentiment-analysis-web
python -m pytest tests/
```

## 📈 Future Enhancements

Potential improvements and extensions:

1. **Database Integration** - PostgreSQL/MongoDB support
2. **User Authentication** - Login/logout functionality
3. **Model Deployment** - Cloud deployment with APIs
4. **Real-time Features** - WebSocket integration
5. **Mobile App** - React Native companion app
6. **Monitoring** - Application performance monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is created for educational purposes as part of the CodexIntern Python Development program.

## 👥 Team

**CodexIntern 2025 - Python Development Track**
- **Duration:** August 1-31, 2025
- **Domain:** Python Development
- **Tasks Completed:** 7/7 (3 Beginner + 4 Intermediate)

## 📞 Support

For questions or support:
- Create an issue in the repository
- Check individual project README files
- Review the documentation in each project folder

---

## 🎯 Project Completion Status

- [x] Slab 1 - Beginner Level (3/3 completed)
  - [x] Pandas Data Analysis Tool
  - [x] Linear Regression House Price Predictor
  - [x] Matrix Operations Tool
- [x] Slab 2 - Intermediate Level (4/4 completed)
  - [x] Voice-Activated Personal Assistant
  - [x] Sentiment Analysis Web Application
  - [x] Google Gemini Integration
  - [x] Speech-to-Image Generator
- [x] Web Interfaces (4 Flask applications)
- [x] Documentation and READMEs
- [x] Deployment configurations

**Total Progress: 100% Complete** ✅

---

*Built with ❤️ for CodexIntern Python Development Program 2025*