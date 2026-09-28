# 🍅 Tomato Leaf Disease Detection

An end-to-end machine learning application for detecting tomato leaf diseases from uploaded images.

The project combines **TensorFlow, FastAPI, Streamlit, Docker, Docker Compose, and automated testing** into a complete ML application pipeline.

## 🚀 Features

* 🧠 Deep-learning-based tomato leaf disease classification
* 🖼️ Image upload through a Streamlit web interface
* ⚡ FastAPI REST API for model inference
* 📊 Prediction confidence and class probabilities
* 🔌 Streamlit ↔ FastAPI integration
* 🐳 Dockerized application
* 🐳 Docker Compose with separate API and frontend services
* ❤️ API health-check endpoint
* 🧪 Automated tests with pytest
* 📦 Configurable API URL through environment variables
* 📁 Organized source-code structure for data, inference, and API components

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │      User           │
                    │   Uploads Image     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Streamlit       │
                    │    Frontend :8501   │
                    └──────────┬──────────┘
                               │
                               │ HTTP Request
                               ▼
                    ┌─────────────────────┐
                    │       FastAPI       │
                    │     Backend :8000   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  TensorFlow Model   │
                    │      Inference      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Disease Prediction  │
                    │ + Confidence Score  │
                    └─────────────────────┘
```

## 🛠️ Technology Stack

| Category                   | Technology          |
| -------------------------- | ------------------- |
| Programming Language       | Python              |
| Machine Learning           | TensorFlow          |
| Data Processing            | NumPy, Pandas       |
| Image Processing           | Pillow, OpenCV      |
| Visualization              | Matplotlib, Seaborn |
| Machine Learning Utilities | Scikit-learn        |
| Frontend                   | Streamlit           |
| Backend                    | FastAPI             |
| API Server                 | Uvicorn             |
| Testing                    | Pytest              |
| Containerization           | Docker              |
| Orchestration              | Docker Compose      |

## 📂 Project Structure

```text
Tomato Leaf Disease Detection/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│
├── models/
│   └── class_names.json
│
├── notebooks/
│   └── 01_dataset_inspection.ipynb
│
├── src/
│   ├── api/
│   │   ├── client.py
│   │   └── main.py
│   │
│   ├── data/
│   │   └── download_dataset.py
│   │
│   └── inference/
│       ├── __init__.py
│       └── predict.py
│
├── tests/
│   ├── test_api.py
│   ├── test_inference.py
│   ├── test_leaf.jpg
│   ├── test_model.py
│   └── test_prediction.py
│
├── .dockerignore
├── .gitignore
├── DockerFile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── README.md
```

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/AKASHRAUT108/Tomato-leaf-disease-detection.git
cd Tomato-leaf-disease-detection
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

```powershell
python -m uvicorn src.api.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

### 5. Start Streamlit

In another terminal:

```powershell
streamlit run app/streamlit_app.py
```

The application will be available at:

```text
http://localhost:8501
```

## 🐳 Run with Docker Compose

Docker Compose is the recommended way to run the complete application.

Build the services:

```powershell
docker compose build
```

Start the application:

```powershell
docker compose up
```

The services are exposed at:

```text
Streamlit: http://localhost:8501
FastAPI:   http://localhost:8000
Swagger:   http://localhost:8000/docs
```

Stop the services:

```powershell
docker compose down
```

### Docker Architecture

Docker Compose runs two services:

```text
┌──────────────────────────────┐
│       Streamlit :8501       │
│        Frontend             │
└──────────────┬───────────────┘
               │
               │ API_URL
               ▼
┌──────────────────────────────┐
│        FastAPI :8000        │
│         Backend             │
└──────────────────────────────┘
```

The Streamlit container communicates with FastAPI using:

```text
http://api:8000
```

The API includes a health check, and Streamlit waits for the API service to become healthy before starting.

## 🔌 API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

### Prediction

```http
POST /predict
```

Upload an image using the `file` form field.

Example using `curl`:

```powershell
curl.exe -X POST "http://localhost:8000/predict" `
  -F "file=@tests/test_leaf.jpg"
```

Example response:

```json
{
  "filename": "test_leaf.jpg",
  "predicted_class": "Late_blight227",
  "confidence": 0.7533348202705383,
  "probabilities": [
    0.02295690029859543,
    0.04417858272790909,
    0.7533348202705383
  ]
}
```

The probability array contains the model's predicted probability distribution across the supported classes.

## 🧪 Testing

Run the test suite with:

```powershell
pytest
```

The project includes tests covering:

* API functionality
* Model functionality
* Inference
* Prediction behavior

A test image is included at:

```text
tests/test_leaf.jpg
```

## 🔍 API Verification

Health check:

```powershell
Invoke-WebRequest http://localhost:8000/health
```

Expected HTTP status:

```text
200 OK
```

Swagger UI:

```text
http://localhost:8000/docs
```

## 📊 Example Prediction

A test request successfully produced:

```text
Predicted class: Late_blight227
Confidence:      75.33%
```

This verifies the complete inference path from image upload through the API to the TensorFlow model.

## 🔐 Configuration

The Streamlit frontend uses the `API_URL` environment variable.

For local development:

```text
API_URL=http://127.0.0.1:8000
```

Inside Docker Compose:

```text
API_URL=http://api:8000
```

This allows the same application code to work in both local and containerized environments.

## 📌 Project Highlights

This project demonstrates an end-to-end machine learning workflow rather than only model training.

It includes:

* Machine learning inference
* Image processing
* REST API development
* Frontend development
* Automated testing
* Environment-based configuration
* Containerization
* Multi-container orchestration
* Service health checks

## 🔮 Future Improvements

Possible future improvements include:

* Model performance optimization
* Additional disease classes
* Model explainability using Grad-CAM
* Prediction history
* Authentication for the API
* Cloud deployment
* CI/CD pipeline
* Monitoring and logging
* Model versioning
* Improved UI and visual analytics

## 👨‍💻 Author

**Akash Raut**

GitHub:

https://github.com/AKASHRAUT108
