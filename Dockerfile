FROM python:3.10

# Create a directory for the app
WORKDIR /imdb_reviews

# Copy the files into the app directory
COPY . /imdb_reviews

# Install dependencies
RUN pip install tensorflow tensorflow-hub tensorflow-datasets pickle-mixin flask

# Expose port 5000
EXPOSE 5000

# Command to run the application
CMD ["python3", "script.py"]

# Build the Docker image
docker build -t sentimentanalysisapp .

# Run the Docker container
docker run -p 5000:5000 sentimentanalysisapp:latest
