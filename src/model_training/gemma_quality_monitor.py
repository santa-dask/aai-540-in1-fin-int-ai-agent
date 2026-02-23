
import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from datasets import Dataset
import pandas as pd
import pandas_gbq
from datetime import datetime
import uuid

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

def calculate_perplexity(text, model, tokenizer, device, max_seq_length=512):
    if not text or not isinstance(text, str): # Handle empty or non-string inputs
        return None
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_length).to(device)
    if inputs['input_ids'].shape[1] < 2: # Need at least 2 tokens to calculate perplexity (one for label)
        return None

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

def main():
    parser = argparse.ArgumentParser(description="Gemma Model Quality Monitoring Script.")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-4b-it", help="Hugging Face model ID to load.")
    parser.add_argument("--lora_adapter_path", type=str, required=True, help="GCS path to LoRA adapters.")
    parser.add_argument("--monitoring_data_path", type=str, required=True, help="GCS path to the monitoring CSV dataset.")
    parser.add_argument("--output_metrics_table", type=str, default="cfpb_analysis.model_quality_metrics", help="BigQuery table ID for storing monitoring metrics.")
    parser.add_argument("--project_id", type=str, default=os.getenv("GOOGLE_CLOUD_PROJECT"), help="GCP project ID.")
    parser.add_argument("--narrative_column", type=str, default="narrative_text", help="Column in monitoring data containing text for evaluation.")

    args = parser.parse_args()

    print(f"Starting Gemma model quality monitoring...")
    print(f"Base Model: {args.model_id}")
    print(f"LoRA Adapters Path: {args.lora_adapter_path}")
    print(f"Monitoring Data Path: {args.monitoring_data_path}")
    print(f"Output BigQuery Table: {args.output_metrics_table}")

    # 1. Set up compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load HF_TOKEN
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set.")
    
    # 3. Load base model and tokenizer
    print(f"Loading base model {args.model_id} and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 

    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, token=hf_token)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.to(device)

    # 4. Load LoRA adapters and merge
    print(f"Loading LoRA adapters from {args.lora_adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
    model = model.merge_and_unload()
    print("LoRA adapters loaded and merged.")
    model.eval() # Set model to evaluation mode

    # 5. Load monitoring dataset from GCS
    print(f"Loading monitoring data from {args.monitoring_data_path}...")
    try:
        monitoring_df = pd.read_csv(args.monitoring_data_path, header=None, names=[args.narrative_column])
        print(f"Loaded {len(monitoring_df)} samples for monitoring.")
    except Exception as e:
        print(f"Error loading monitoring data: {e}")
        return

    # 6. Calculate perplexity for each sample
    print("Calculating perplexity scores...")
    metrics_records = []
    current_run_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    for index, row in monitoring_df.iterrows():
        text_sample = row[args.narrative_column]
        perplexity = calculate_perplexity(text_sample, model, tokenizer, device)
        if perplexity is not None:
            metrics_records.append({
                'timestamp': timestamp,
                'run_id': current_run_id,
                'metric_name': 'perplexity',
                'metric_value': perplexity,
                'text_sample': text_sample
            })
        else:
            print(f"Skipping perplexity calculation for sample {index} due to invalid text.")

    if not metrics_records:
        print("No valid perplexity scores calculated. Exiting.")
        return

    # 7. Create pandas DataFrame from collected metrics
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
    
    print(f"Collected {len(metrics_df)} perplexity scores.")
    print("First 5 rows of metrics_df:")
    print(metrics_df.head())

    # 8. Save metrics DataFrame to BigQuery
    print(f"Saving metrics to BigQuery table: {args.output_metrics_table}...")
    try:
        pandas_gbq.to_gbq(
            metrics_df,
            destination_table=args.output_metrics_table,
            project_id=args.project_id,
            if_exists='append', # Append to the table for historical tracking
            table_schema=[
                {'name': 'timestamp', 'type': 'TIMESTAMP'},
                {'name': 'run_id', 'type': 'STRING'},
                {'name': 'metric_name', 'type': 'STRING'},
                {'name': 'metric_value', 'type': 'FLOAT'},
                {'name': 'text_sample', 'type': 'STRING'}
            ]
        )
        print("Metrics successfully saved to BigQuery.")
    except Exception as e:
        print(f"Error saving metrics to BigQuery: {e}")

if __name__ == "__main__":
    main()
