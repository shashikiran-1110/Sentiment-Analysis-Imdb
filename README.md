# Sentiment Analaysis on IMDB Movie Reviews Deployment Pipeline

## Source Code
- **preprocessing.py**: Contains code for developing the AI model.
- **script.py**: Includes code for the web service.
- **Dockerfile**: Specifies the containerization of the AI model and web service.
- **deployment.yaml**: Kubernetes deployment configuration.

## Usage
1. **AI Model Development**
    - Execute `preprocessing.py` for model development.

2. **Web Service Creation**
    - Run `script.py` to start the web service.
    - API endpoint: `/predict`

3. **Containerization with Docker**
    - Build Docker image: `docker build -t sentimentanalysisapp .`
    - Run Docker container: `docker run -p 5000:5000 sentimentanalysisapp:latest`

4. **Deployment with Kubernetes**
    - Apply Kubernetes deployment: `kubectl apply -f deployment.yaml`
    - Check pod status: `kubectl get pods`

## Notes
- Ensure dependencies (e.g., TensorFlow, Flask) are installed.
- Adjust ports or configurations if needed.
