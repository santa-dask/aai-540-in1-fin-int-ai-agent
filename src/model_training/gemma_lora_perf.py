import argparse
import os
import sys
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import Dataset
from google.cloud import bigquery, storage
import pandas as pd

# Add source path to sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

def main():
    parser = argparse.ArgumentParser(description="LoRA Model Perplexity Calculation for Gemma 3.")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-4b-it", help="Base model ID.")
    parser.add_argument("--lora_adapter_path", type=str, required=True, help="Path to LoRA adapters (gcsfuse mount).")
    parser.add_argument("--bigquery_table_id", type=str, default="cfpb_analysis.complaints_features", help="BQ table ID.")
    parser.add_argument("--narrative_column", type=str, default="narrative_text", help="Column name.")
    parser.add_argument("--output_path", type=str, required=True, help="Output GCS path (e.g. gs://bucket/result.txt).")
    parser.add_argument("--project_id", type=str, default=os.getenv("GOOGLE_CLOUD_PROJECT"), help="GCP project ID.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")

    args = parser.parse_args()
    hf_token = os.getenv("HF_TOKEN")

    # 1. Load data from BigQuery using official client
    print(f" Loading data from BigQuery: {args.project_id}.{args.bigquery_table_id}")
    try:
        bq_client = bigquery.Client(project=args.project_id)
        query = f"SELECT {args.narrative_column} FROM `{args.project_id}.{args.bigquery_table_id}` WHERE {args.narrative_column} IS NOT NULL LIMIT 100"
        df = bq_client.query(query).to_dataframe()
        dataset = Dataset.from_pandas(df)
        print(f"Loaded {len(dataset)} examples.")
    except Exception as e:
        print(f" BigQuery Error: {e}")
        return

    # 2. Load base model and tokenizer
    print(f" Loading base model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    
    # Gemma 3 Standard Padding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        device_map="auto"
    )

    # 3. Load and Merge LoRA Adapters
    # Adjust path: handle both raw bucket name and full gs:// path
    local_lora_path = args.lora_adapter_path.replace("gs://", "/gcs/")
    print(f" Merging adapters from: {local_lora_path}")
    
    try:
        model = PeftModel.from_pretrained(model, local_lora_path)
        model = model.merge_and_unload()
        model.eval()
        print(" LoRA adapters merged successfully.")
    except Exception as e:
        print(f" Error merging adapters: {e}. Ensure gcsfuse is mounted.")
        return

    # 4. Tokenize with Gemma 3 Token Type IDs
    print("Tokenizing data...")
    def tokenize_function(examples):
        return tokenizer(
            examples[args.narrative_column], 
            truncation=True, 
            max_length=args.max_seq_length, 
            padding="max_length",
            return_token_type_ids=True 
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # 5. Calculate Perplexity
    print("Calculating perplexity...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_loss = 0
    total_tokens = 0

    for i in range(0, len(tokenized_dataset), args.batch_size):
        batch = tokenized_dataset[i : i + args.batch_size]
        
        # Prepare inputs including token_type_ids
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        token_type_ids = torch.tensor(batch["token_type_ids"]).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids, 
                labels=input_ids
            )
            # CrossEntropyLoss returns mean loss per non-ignored token
            loss = outputs.loss
            
            # Count only active (non-padded) tokens
            num_active_tokens = attention_mask.sum().item()
            total_loss += (loss.item() * num_active_tokens)
            total_tokens += num_active_tokens

    if total_tokens == 0:
        perplexity_score = float('inf')
    else:
        avg_loss = total_loss / total_tokens
        perplexity_score = math.exp(avg_loss)

    print(f" Perplexity Score: {perplexity_score}")

    # 6. Save results directly via gcsfuse or standard Client
    local_file = "gemma_lora_perplexity.txt"
    with open(local_file, "w") as f:
        f.write(f"Perplexity Score for {args.model_id} + LoRA: {perplexity_score}\n")

    # Final Upload logic
    try:
        # If output_path is a gcsfuse mount path, just move it
        if args.output_path.startswith("/gcs/"):
            import shutil
            shutil.copy(local_file, args.output_path)
        else:
            # Otherwise use storage client for gs:// paths
            bucket_name = args.output_path.split('/')[2]
            blob_name = '/'.join(args.output_path.split('/')[3:])
            storage_client = storage.Client(project=args.project_id)
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file)
        print(f" Results saved to {args.output_path}")
    except Exception as e:
        print(f" Error saving results: {e}")

if __name__ == "__main__":
    main()