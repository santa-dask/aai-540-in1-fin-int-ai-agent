
import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import fsspec

# Add src directory to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

def main():
    parser = argparse.ArgumentParser(description="Save Merged LoRA Model Script.")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-4b-it", help="Hugging Face model ID to load.")
    parser.add_argument("--lora_adapter_gcs_path", type=str, required=True, help="GCS path to LoRA adapters.")
    parser.add_argument("--output_local_dir", type=str, required=True, help="Local directory to save the merged model.")
    parser.add_argument("--project_id", type=str, default=os.getenv("GOOGLE_CLOUD_PROJECT"), help="GCP project ID.")

    args = parser.parse_args()

    print(f"Starting merging LoRA adapters into base model: {args.model_id}")
    print(f"LoRA Adapters GCS Path: {args.lora_adapter_gcs_path}")
    print(f"Output Local Directory: {args.output_local_dir}")

    # 1. Ensure HF_TOKEN is available
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set.")

    # 2. Create a local temporary directory for downloading adapters
    temp_lora_dir = "./temp_lora_adapters"
    os.makedirs(temp_lora_dir, exist_ok=True)

    # Use fsspec for GCS operations
    fs = fsspec.filesystem('gs')

    # 3. Download LoRA adapters from GCS
    print(f"Downloading LoRA adapters from {args.lora_adapter_gcs_path} to {temp_lora_dir}...")
    fs.get(args.lora_adapter_gcs_path, temp_lora_dir, recursive=True)
    print("LoRA adapters downloaded successfully.")

    # 4. Load base model and tokenizer
    print(f"Loading base model {args.model_id} and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # or use tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, token=hf_token)
    base_model.resize_token_embeddings(len(tokenizer))

    # 5. Load LoRA adapters and merge
    print(f"Loading LoRA adapters from local path {temp_lora_dir} and merging...")
    model = PeftModel.from_pretrained(base_model, temp_lora_dir)
    merged_model = model.merge_and_unload()
    print("LoRA adapters merged successfully.")

    # 6. Save the merged model and tokenizer locally
    print(f"Saving merged model and tokenizer to {args.output_local_dir}...")
    os.makedirs(args.output_local_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_local_dir)
    tokenizer.save_pretrained(args.output_local_dir)
    print("Merged model and tokenizer saved successfully.")

    # Clean up temporary LoRA directory
    os.system(f"rm -rf {temp_lora_dir}")
    print(f"Cleaned up temporary directory: {temp_lora_dir}")

if __name__ == "__main__":
    main()
