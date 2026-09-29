Tomato Leaf Disease Detection

An end-to-end deep learning application that detects tomato leaf diseases from images using a fine-tuned MobileNetV2 model.

The project goes beyond model training by integrating the trained model into a complete application with a Streamlit user interface, FastAPI backend, Docker containerization, Docker Compose, and cloud deployment on Render.

The goal was to build a complete machine learning application that can take an image from a user, process it through a trained deep learning model, and return an easy-to-understand disease prediction.

Live Demo
Web Application

Try the application:

https://tomato-leaf-disease-detection-1-wkc6.onrender.com

Upload a tomato leaf image and receive a predicted disease class along with confidence and prediction probabilities.

FastAPI Backend

https://tomato-leaf-disease-detection-bqtc.onrender.com

Interactive API Documentation

https://tomato-leaf-disease-detection-bqtc.onrender.com/docs

API Health Check

https://tomato-leaf-disease-detection-bqtc.onrender.com/health

Project Overview

Plant diseases can affect crop health and productivity, and identifying symptoms from leaf images can be a useful computer vision problem.

This project explores how deep learning can be used to classify tomato leaf images into disease categories.

Instead of stopping after training a classification model, the project follows the complete machine learning application lifecycle:

Image
  |
  v
Image Preprocessing
  |
  v
Fine-Tuned MobileNetV2
  |
  v
Disease Classification
  |
  v
Prediction + Confidence
  |
  v
FastAPI Backend
  |
  v
Streamlit Application
  |
  v
Docker
  |
  v
Render Deployment
Problem Statement

The objective of this project is to build a computer vision application capable of analyzing a tomato leaf image and predicting the corresponding disease class.

The system should provide a simple interface where a user can:

Upload a tomato leaf image.
Send the image to the prediction system.
Process the image using the trained model.
Receive the predicted disease.
View the prediction confidence and probability information.
Project Objective

The main objectives of the project were:

Build a deep learning image classification model.
Apply transfer learning using MobileNetV2.
Create a reusable inference pipeline.
Expose the model through a REST API.
Build a user-friendly Streamlit interface.
Containerize the application using Docker.
Run the frontend and backend using Docker Compose.
Deploy the application to the cloud.
Create a complete end-to-end machine learning project suitable for real-world demonstration.
How the Application Works

The application follows a simple prediction workflow.

Step 1 - Upload Image

The user uploads a tomato leaf image through the Streamlit interface.

Supported formats include:

JPG
JPEG
PNG
Step 2 - Image Validation

The backend checks:

File type
Image validity
Image dimensions

Invalid or unsupported images are rejected before prediction.

Step 3 - Image Preprocessing

The uploaded image is:

Converted to RGB
Resized to 224 x 224 pixels
Converted into a NumPy array
Preprocessed using MobileNetV2 preprocessing
Step 4 - Model Inference

The processed image is passed to the fine-tuned MobileNetV2 model.

The model generates probabilities for the supported disease classes.

Step 5 - Prediction

The class with the highest predicted probability is selected as the final prediction.

The API returns:

Filename
Predicted class
Confidence
Prediction probabilities
Step 6 - Display Results

The Streamlit application presents the prediction in a user-friendly format.

It also provides additional prediction information such as top predictions, confidence visualization, and prediction history.

Key Features
Deep Learning
Fine-tuned MobileNetV2 model
Image classification
Transfer learning
MobileNetV2 preprocessing
Probability-based predictions
Streamlit Application

The frontend provides:

Image upload
Disease prediction
Confidence display
Top predictions
Probability distribution
Prediction history
Prediction statistics
Confidence trend visualization
Prediction history visualization
CSV download
FastAPI Backend

The API provides:

REST API
Image upload endpoint
Prediction endpoint
Health check endpoint
Input validation
Error handling
Automatic Swagger documentation
Docker

The project includes:

Docker image configuration
Docker Compose
API container
Streamlit container
Service-to-service communication
API health checks
Container dependency management
Cloud Deployment

The application is deployed with separate frontend and backend services on Render.

Machine Learning Approach

The project uses transfer learning with MobileNetV2.

MobileNetV2 provides a lightweight convolutional neural network architecture that can be adapted to image classification tasks.

The model used in this project is a fine-tuned MobileNetV2 model trained for tomato leaf disease classification.

Model Pipeline
Input Image
     |
     v
RGB Conversion
     |
     v
Resize to 224 x 224
     |
     v
MobileNetV2 Preprocessing
     |
     v
Fine-Tuned MobileNetV2
     |
     v
Class Probabilities
     |
     v
Predicted Disease
Model Input
Image format: RGB
Input size: 224 x 224
Model: MobileNetV2
Model Output

The inference pipeline returns:

{
  "class": "Predicted disease class",
  "confidence": 0.75,
  "probabilities": []
}
Example Prediction

A production API test returned a prediction in the following format:

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

The probability values represent the model's predicted probability distribution across the supported classes.

Application Architecture

The project separates the user interface, API, and machine learning inference logic.

                         USER
                           |
                           v
                 +-------------------+
                 | Streamlit Frontend|
                 +---------+---------+
                           |
                           | HTTP Request
                           v
                 +-------------------+
                 |   FastAPI API     |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 | Image Validation  |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 | Image Preprocess  |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 |  MobileNetV2      |
                 |  Fine-Tuned Model |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 | Prediction Result |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 | Streamlit Display |
                 +-------------------+
Deployment Architecture

The deployed system uses separate frontend and backend services.

                         Internet
                            |
              +-------------+-------------+
              |                           |
              v                           v
      Streamlit Frontend          FastAPI Backend
          Render                     Render
              |                           |
              |       HTTP Request        |
              +---------------------------+
                          |
                          v
                  MobileNetV2 Model
                          |
                          v
                  Disease Prediction
Technology Stack
Area	Technology
Programming	Python
Deep Learning	TensorFlow / Keras
Model	MobileNetV2
Image Processing	Pillow
Numerical Computing	NumPy
Backend	FastAPI
API Server	Uvicorn
Frontend	Streamlit
HTTP Client	Requests
Containerization	Docker
Multi-container Setup	Docker Compose
Version Control	Git / GitHub
Cloud Deployment	Render
Project Structure
Tomato-leaf-disease-detection/
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
API

The FastAPI backend exposes three main endpoints.

Root Endpoint
GET /

Provides basic information about the API.

Health Check
GET /health

Example response:

{
  "status": "healthy"
}
Prediction
POST /predict

The endpoint accepts an image through the file field.

Example:

curl.exe -X POST "http://localhost:8000/predict" -F "file=@tests/test_leaf.jpg"

The production API is available at:

https://tomato-leaf-disease-detection-bqtc.onrender.com

Swagger Documentation

FastAPI automatically generates interactive API documentation.

Local

http://localhost:8000/docs

Production

https://tomato-leaf-disease-detection-bqtc.onrender.com/docs

The Swagger interface makes it possible to test the API without writing a separate client.

Running the Project Locally
1. Clone the Repository
git clone https://github.com/AKASHRAUT108/Tomato-leaf-disease-detection.git

cd Tomato-leaf-disease-detection
2. Create a Virtual Environment
python -m venv .venv

Activate it on Windows PowerShell:

.venv\Scripts\Activate.ps1
3. Install Dependencies
pip install -r requirements.txt
4. Start the FastAPI Backend
python -m uvicorn src.api.main:app --reload

The API will be available at:

http://127.0.0.1:8000

Swagger:

http://127.0.0.1:8000/docs
5. Start Streamlit

Open another terminal with the virtual environment activated:

streamlit run app/streamlit_app.py

The application will be available at:

http://localhost:8501
Running with Docker

The project can also be run using Docker Compose.

Build the Containers
docker compose build
Start the Application
docker compose up

The services run on:

FastAPI   -> http://localhost:8000
Streamlit -> http://localhost:8501
Stop the Application
docker compose down

Docker Compose also includes an API health check so that the Streamlit service can wait for the backend to become healthy.

Deployment

The application is deployed on Render using containerized services.

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

The Streamlit frontend communicates with the deployed FastAPI backend through the configured API_URL environment variable.

What I Learned

Building this project helped me understand how a machine learning model can be turned into a complete application.

Some of the main areas I worked with include:

Deep learning for image classification
Transfer learning
MobileNetV2
Image preprocessing
TensorFlow and Keras
Model inference
FastAPI REST APIs
Streamlit application development
API-to-frontend communication
Docker containerization
Docker Compose
Health checks and service dependencies
Cloud deployment
Git and GitHub
Structuring a machine learning project for deployment

The biggest learning from the project was understanding that building a machine learning project is not only about training a model.

A complete project also needs:

Model
  +
Inference Pipeline
  +
Backend API
  +
Frontend
  +
Containerization
  +
Deployment
Future Improvements

Possible improvements for future versions include:

Adding more detailed model evaluation reports
Improving prediction explanations
Adding additional image augmentation strategies
Adding model version management
Adding automated testing
Adding CI/CD through GitHub Actions
Adding monitoring for the deployed API
Improving the frontend user experience
Adding more agricultural information around predicted diseases
Disclaimer

This project is intended for educational and demonstration purposes.

The predictions generated by the application should not be treated as a replacement for professional agricultural diagnosis or expert advice.

Author

Akash Raut

GitHub:

https://github.com/AKASHRAUT108

Project Repository:

https://github.com/AKASHRAUT108/Tomato-leaf-disease-detection

Try the Application

If you would like to see the project in action:

Live Application

https://tomato-leaf-disease-detection-1-wkc6.onrender.com

API

https://tomato-leaf-disease-detection-bqtc.onrender.com

Swagger Documentation

https://tomato-leaf-disease-detection-bqtc.onrender.com/docs