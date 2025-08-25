# 🩺 Diabetes Risk Chatbot with MCP Integration - Repository Summary

## 🚀 Project Overview
A comprehensive, bilingual (English/Arabic) diabetes risk assessment system that combines machine learning, MCP (Model Context Protocol) tools, and conversational AI to provide personalized health insights and recommendations.

## ✨ Key Features

### 🤖 **AI-Powered Risk Assessment**
- **Multi-Model ML Pipeline**: Logistic Regression, Random Forest, XGBoost, SVM, MLP
- **SHAP Explainability**: Global and local feature importance analysis
- **Partial Dependence Plots**: Visual risk factor analysis
- **Threshold Optimization**: Precision-recall balanced predictions

### 🛠️ **MCP Server Integration**
- **Lab Results Tool**: `labs.getLatestHbA1c` - Fetch latest lab values
- **Guidelines Tool**: `guidelines.lookup` - Access clinical guidelines
- **Genomics Tool**: `genomics.lookup` - Genetic variant analysis
- **Translation Tool**: `translate.detectAndTranslate` - Multilingual support

### 💬 **Conversational AI Layer**
- **Bilingual Support**: English and Arabic with RTL layout
- **Context Awareness**: Remembers patient history and preferences
- **Sentiment Analysis**: Adapts tone based on user emotional state
- **Behavioral Coaching**: Personalized lifestyle recommendations

### 📱 **Modern Web Interface**
- **Responsive Design**: Mobile-first, professional UI
- **Real-time Charts**: Risk history visualization with Chart.js
- **Multi-modal Input**: Voice transcription, lab uploads, wearable data
- **Interactive Cards**: Structured input sections with validation

### 📊 **Advanced Analytics**
- **Risk Tracking**: Historical risk score monitoring
- **Factor Analysis**: Top contributing risk factors
- **Personalized Reports**: PDF generation with charts and recommendations
- **Population Insights**: Ethnicity-specific risk considerations

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Web UI  │    │  Chat Service   │    │  Model Service  │
│   (EN/AR + RTL) │◄──►│   (Node.js)     │◄──►│   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   MCP Server    │    │   ML Pipeline   │
                       │   (FastAPI)     │    │  (scikit-learn) │
                       └─────────────────┘    └─────────────────┘
```

## 🎯 **Objectives Completed**

- ✅ **Objective 1**: Data Collection & EDA
- ✅ **Objective 2**: ML Training & Model Selection  
- ✅ **Objective 3**: Model Service API
- ✅ **Objective 4**: MCP Server Tools
- ✅ **Objective 5**: Conversational Layer
- ✅ **Objective 6**: Web Interface
- ✅ **Objective 7**: Multi-Modal Input
- ✅ **Objective 8**: Explainability
- ✅ **Objective 9**: Precision Medicine
- ✅ **Objective 10**: Behavioral Coaching
- ✅ **Objective 11**: Clinician Reporting

## 🛠️ **Technology Stack**

### **Backend Services**
- **Python**: FastAPI, scikit-learn, XGBoost, SHAP
- **Node.js**: Express, OpenAI Whisper, Puppeteer
- **ML**: Logistic Regression, Random Forest, XGBoost, SVM, MLP

### **Frontend**
- **React 18**: Modern hooks and functional components
- **Vite**: Fast development and build tooling
- **Chart.js**: Interactive data visualization
- **CSS3**: Advanced animations and gradients

### **AI/ML**
- **OpenAI**: GPT-4o-mini, Whisper ASR
- **SHAP**: Model explainability and feature importance
- **Scikit-learn**: Comprehensive ML toolkit

### **DevOps & Testing**
- **Jest**: Unit testing framework
- **Puppeteer**: Automated screenshots and PDF generation
- **Git**: Version control with comprehensive .gitignore

## 📁 **Repository Structure**

```
├── 📊 ml/                    # Machine Learning Pipeline
├── 🚀 model_service/         # FastAPI Model Service
├── 🛠️ mcp_server/           # MCP Tools Server
├── 💬 chat_service/          # Node.js Chat Backend
├── 🌐 web/                   # React Web Application
├── 📚 docs/                  # Project Documentation
├── 📈 artifacts/             # ML Models & Visualizations
├── 📋 reports/               # Jupyter Notebooks
└── 📸 slides/                # Presentation Materials
```

## 🚀 **Quick Start**

### **1. Clone Repository**
```bash
git clone https://github.com/jaannawaz/Diabetes-Risk-Chatbot-with-MCP-Integration.git
cd Diabetes-Risk-Chatbot-with-MCP-Integration
```

### **2. Setup Environment**
```bash
# Python Environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r model_service/requirements.txt
pip install -r mcp_server/requirements.txt

# Node.js Dependencies
cd chat_service && npm install
cd ../web && npm install
```

### **3. Run Services**
```bash
# Terminal 1: Model Service
source .venv/bin/activate
uvicorn model_service.app:app --port 8001

# Terminal 2: MCP Server  
source .venv/bin/activate
uvicorn mcp_server.app:app --port 8002

# Terminal 3: Chat Service
cd chat_service
MODEL_URL=http://127.0.0.1:8001 MCP_URL=http://127.0.0.1:8002 npm start

# Terminal 4: Web UI
cd web
npm run dev
```

### **4. Access Application**
- **Web UI**: http://localhost:5173
- **Model API**: http://localhost:8001/docs
- **MCP Server**: http://localhost:8002/docs
- **Chat API**: http://localhost:8003

## 🔑 **Environment Variables**

Create `.env` files in respective service directories:

```bash
# chat_service/.env
OPENAI_API_KEY=your_openai_key
GROK_API_KEY=your_grok_key  # Optional

# model_service/.env
MODEL_PATH=artifacts/best_model.joblib
```

## 📊 **Dataset Information**

- **Source**: Diabetes Prediction Dataset
- **Features**: 15 clinical and demographic variables
- **Target**: Binary diabetes risk classification
- **Size**: ~1000+ samples with balanced classes
- **License**: Open source for research purposes

## 🎨 **UI/UX Features**

- **Modern Design**: Glass morphism with gradient backgrounds
- **Responsive Layout**: Two-panel design (input + chat)
- **Interactive Elements**: Hover effects, smooth animations
- **Accessibility**: RTL support, keyboard navigation
- **Professional Look**: Medical-grade interface design

## 🔬 **Research & Development**

- **Model Validation**: Comprehensive evaluation metrics
- **Feature Engineering**: Domain-specific transformations
- **Explainability**: SHAP analysis for clinical interpretability
- **Performance**: ROC-AUC, precision-recall optimization
- **Scalability**: Microservices architecture

## 🤝 **Contributing**

This is a research and demonstration project. Contributions are welcome for:
- Bug fixes and improvements
- Additional ML models
- Enhanced UI components
- Documentation updates
- Testing and validation

## 📄 **License**

This project is for educational and research purposes. Please ensure compliance with:
- Medical device regulations
- Data privacy laws (HIPAA, GDPR)
- Ethical AI guidelines
- Clinical validation requirements

## 🌟 **Highlights**

- **First-of-its-kind**: MCP integration in healthcare chatbot
- **Bilingual Support**: English/Arabic with cultural considerations
- **Clinical Grade**: Explainable AI with medical disclaimers
- **Modern Stack**: Latest technologies and best practices
- **Comprehensive**: End-to-end ML pipeline to production UI

---

**Built with ❤️ for advancing healthcare AI and improving patient outcomes**

*For questions or collaboration, please open an issue or discussion on GitHub.*
