#!/bin/bash
# Build and push the Docker image to Google Artifact Registry

# Ensure the script stops if any command fails
set -e

# Navigate to the project directory
cd "${PROJECT_DIR}"

# Build the Docker image and tag it
echo "Building Docker image: ${ML_IMAGE_PATH}"
gcloud builds submit . --tag "${ML_IMAGE_PATH}"

echo "Docker image build and push complete."

