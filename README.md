# Tomato Leaf Disease Detection

A deep learning application that helps identify tomato leaf diseases from images.

This project takes a tomato leaf image, processes it, and predicts the most likely disease using a fine-tuned MobileNetV2 model. The trained model is connected to a FastAPI backend and a Streamlit frontend, then packaged with Docker and deployed using Render.

## Live Project

### Deployed Application

Try the project here:

https://tomato-leaf-disease-detection-1-wkc6.onrender.com

You can upload a tomato leaf image and get a disease prediction along with its confidence and probability information.

### Backend API

https://tomato-leaf-disease-detection-bqtc.onrender.com

### Swagger API Documentation

https://tomato-leaf-disease-detection-bqtc.onrender.com/docs

### API Health Check

https://tomato-leaf-disease-detection-bqtc.onrender.com/health

Expected response:

```json
{
  "status": "healthy"
}

About the Project

Tomato Leaf Disease Detection is a computer vision and deep learning project built to classify tomato leaf images into different disease categories.

The main goal of the project was not only to train a machine learning model, but also to turn that model into a complete application that people can actually use.

The project includes:

A trained deep learning model
Image preprocessing
FastAPI backend
Streamlit frontend
Docker containerization
Docker Compose
API health checks
Cloud deployment using Render
Git and GitHub version control

The complete workflow is:

User uploads image
        |
        v
Streamlit frontend
        |
        | HTTP request
        v
FastAPI backend
        |
        v
Image preprocessing
        |
        v
Fine-tuned MobileNetV2
        |
        v
Disease prediction
        |
        v
Confidence and probabilities
        |
        v
Result displayed to user
Features
Image Classification

The application allows users to:

Upload tomato leaf images
Use JPG, JPEG, or PNG files
Validate uploaded images
Check minimum image dimensions
Convert images to RGB
Resize images to 224 x 224 pixels
Apply MobileNetV2 preprocessing
Predict the disease class
Prediction Results

The Streamlit application provides:

Predicted disease
Prediction confidence
Top predictions
Probability distribution
Prediction history
Prediction statistics
Most frequently predicted disease
Disease prediction counts
Confidence trend visualization
Prediction history visualization
CSV download of prediction history
FastAPI Backend

The backend provides:

REST API
Health check endpoint
Image prediction endpoint
Automatic Swagger documentation
Image validation
Error handling
Docker

The project supports:

Docker image builds
Docker Compose
Separate API and Streamlit services
Containerized model inference
Service-to-service communication
API health checks
Service dependency management
Cloud Deployment

The application is deployed on Render with:

Streamlit frontend
FastAPI backend
Docker-based deployment
Public API
Public web application
Machine Learning

The project uses a fine-tuned MobileNetV2 convolutional neural network for tomato leaf disease classification.

Model
MobileNetV2
     |
     v
Fine-tuning
     |
     v
Tomato Leaf Disease Classification
Input Processing

Before prediction, the uploaded image is:

Converted to RGB
Resized to 224 x 224 pixels
Converted into a NumPy array
Preprocessed using MobileNetV2 preprocessing
Passed to the trained model
Model Output

The model returns:

Predicted disease class
Prediction confidence
Probability for each supported class

Example:

{
  "filename": "test_leaf.jpg",
  "predicted_class": "Late_blight227",
  "confidence": 0.7533
}
System Architecture
                         User
                           |
                           v
                +---------------------+
                | Streamlit Frontend  |
                +----------+----------+
                           |
                           | HTTP
                           v
                +---------------------+
                |   FastAPI Backend   |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Image Preprocessing |
                +----------+----------+
                           |
                           v
                +---------------------+
                |     MobileNetV2     |
                |   Fine-tuned Model  |
                +----------+----------+
                           |
                           v
                +---------------------+
                |  Disease Prediction |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Prediction Results  |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Streamlit Display   |
                +---------------------+
Technology Stack
Category	Technology
Programming Language	Python
Deep Learning	TensorFlow / Keras
Model	MobileNetV2
Numerical Processing	NumPy
Image Processing	Pillow
Backend API	FastAPI
API Server	Uvicorn
Frontend	Streamlit
Containerization	Docker
Multi-service Setup	Docker Compose
Cloud Deployment	Render
Version Control	Git / GitHub
Project Structure
Tomato Leaf Disease Detection/
|
+-- app/
|   +-- streamlit_app.py
|
+-- models/
|   +-- class_names.json
|   +-- mobilenetv2_finetuned_best.keras
|
+-- src/
|   +-- api/
|   |   +-- client.py
|   |   +-- main.py
|   |
|   +-- inference/
|       +-- predict.py
|
+-- tests/
|   +-- test_leaf.jpg
|
+-- DockerFile
+-- docker-compose.yml
+-- .dockerignore
+-- .gitignore
+-- requirements.txt
+-- README.md
API Endpoints
Root
GET /

Returns basic information about the API.

Health Check
GET /health

Example response:

{
  "status": "healthy"
}
Prediction
POST /predict

Upload a tomato leaf image using the file field.

Supported formats:

JPG
JPEG
PNG

Example:

curl.exe -X POST "http://localhost:8000/predict" -F "file=@tests/test_leaf.jpg"

Example response:

{
  "filename": "test_leaf.jpg",
  "predicted_class": "Late_blight227",
  "confidence": 0.7533,
  "probabilities": [
    0.0229,
    0.0441,
    0.7533,
    0.0101,
    0.0292,
    0.0014,
    0.0947,
    0.0045,
    0.0048,
    0.0349
  ]
}

Production API:

https://tomato-leaf-disease-detection-bqtc.onrender.com

Swagger API Documentation

FastAPI automatically provides interactive API documentation.

Local Swagger
http://localhost:8000/docs
Production Swagger

https://tomato-leaf-disease-detection-bqtc.onrender.com/docs

The Swagger interface can be used to test the API endpoints directly.

Run with Docker
Build the containers
docker compose build
Start the application
docker compose up

The project runs two services:

FastAPI   -> http://localhost:8000
Streamlit -> http://localhost:8501
Streamlit
http://localhost:8501
FastAPI
http://localhost:8000
Swagger
http://localhost:8000/docs
Stop the containers
docker compose down
Run Locally Without Docker
1. Clone the repository
git clone https://github.com/AKASHRAUT108/Tomato-leaf-disease-detection.git
cd Tomato-leaf-disease-detection
2. Create a virtual environment

Windows PowerShell:

python -m venv .venv

Activate it:

.venv\Scripts\Activate.ps1
3. Install dependencies
pip install -r requirements.txt
4. Start FastAPI
python -m uvicorn src.api.main:app --reload

API:

http://127.0.0.1:8000

Swagger:

http://127.0.0.1:8000/docs
5. Start Streamlit

Open another terminal with the virtual environment activated:

streamlit run app/streamlit_app.py

Application:

http://localhost:8501
API Testing
Local health check
Invoke-WebRequest "http://localhost:8000/health"
Local prediction
curl.exe -X POST "http://localhost:8000/predict" `
  -F "file=@tests/test_leaf.jpg"
Production health check
Invoke-WebRequest "https://tomato-leaf-disease-detection-bqtc.onrender.com/health"
Deployment

The application is containerized using Docker and deployed using Render.

Frontend

Streamlit:

https://tomato-leaf-disease-detection-1-wkc6.onrender.com

Backend

FastAPI:

https://tomato-leaf-disease-detection-bqtc.onrender.com

API Documentation

https://tomato-leaf-disease-detection-bqtc.onrender.com/docs

Health Check

https://tomato-leaf-disease-detection-bqtc.onrender.com/health

The Streamlit frontend communicates with the FastAPI backend using the API_URL environment variable.

Deployment Architecture
GitHub Repository
        |
        v
   Docker Build
        |
        +---------------------+
        |                     |
        v                     v
 FastAPI Service       Streamlit Service
     Render                  Render
        |                     |
        +------- HTTP --------+
                  |
                  v
          MobileNetV2 Model
                  |
                  v
         Disease Prediction
Project Highlights

This project gave me practical experience with the complete machine learning application workflow.

It covers:

Computer vision
Deep learning
Transfer learning
TensorFlow and Keras
MobileNetV2
Image preprocessing
Model inference
REST API development
FastAPI
Streamlit
Docker
Docker Compose
Cloud deployment
Git and GitHub
Production-style application architecture

The main focus of the project was to go beyond simply training a model. The trained model has been integrated into a complete application with a frontend, backend API, containerized environment, and public cloud deployment.

Disclaimer

This project is intended for educational and demonstration purposes.

The predictions generated by the model should not be considered a substitute for professional agricultural diagnosis or expert advice.

Author

Akash Raut

GitHub:

https://github.com/AKASHRAUT108

Project Repository:

https://github.com/AKASHRAUT108/Tomato-leaf-disease-detection

Try the Application

Live Application:

https://tomato-leaf-disease-detection-1-wkc6.onrender.com

FastAPI Backend:

https://tomato-leaf-disease-detection-bqtc.onrender.com

Swagger Documentation:

https://tomato-leaf-disease-detection-bqtc.onrender.com/docs


