# Use NVIDIA's optimized PyTorch image with Python 3.12 support
#FROM nvcr.io/nvidia/pytorch:24.11-py3
FROM nvcr.io/nvidia/pytorch:26.01-py3
#https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch?version=26.01-py3

WORKDIR /app

# 1. Copy and Install dependencies first (for caching)
COPY setup/server_requirement.txt ./setup/
#RUN pip uninstall -y transformers accelerate peft && pip install --no-cache-dir --force-reinstall -r setup/requirement2.txt
#RUN pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cu128
#RUN pip install --no-cache-dir packaging ninja

#RUN pip install --no-cache-dir \
#    transformers==4.48.3 \
#    accelerate==1.3.0 \
#    peft==0.14.0 \
#    datasets \
#    bitsandbytes \
#    huggingface_hub \
#    gcsfs \
#    fsspec \
#    google-cloud-storage \
#    google-cloud-bigquery \
#    google-cloud-aiplatform \
#    bigframes \
#    pyyaml \
#    tqdm \
#    python-dotenv
RUN pip install --no-cache-dir --force-reinstall -r setup/server_requirement.txt    
RUN pip install --upgrade torch torchvision torchaudio


# 2. Copy requested folders and files
COPY config/ ./config/
COPY src/ ./src/
COPY setup/ ./setup/
#COPY data/ ./data/
COPY job/ ./job/
COPY Dockerfile  ./

# Set Python to flush output immediately for better logging
ENV PYTHONUNBUFFERED=1


# Entry point will be defined in the GKE command (gke_job.yaml)
# CMD ["python3", "src/model_training/training.py"]
