
import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from google.cloud import bigquery
import pandas as pd
import pandas_gbq

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

def main():
    parser = argparse.ArgumentParser(description="LoRA Pretraining Script for Gemma model.")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-4b-it", help="Hugging Face model ID to load.")
    parser.add_argument("--output_dir", type=str, required=True, help="GCS path to save LoRA adapters.")
    parser.add_argument("--bigquery_table_id", type=str, default="cfpb_analysis.complaints_features", help="BigQuery table ID for the training data.")
    parser.add_argument("--narrative_column", type=str, default="narrative_text", help="Column in BigQuery table containing text for pretraining.")
    parser.add_argument("--project_id", type=str, default=os.getenv("GOOGLE_CLOUD_PROJECT"), help="GCP project ID.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for training.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenizer.")

    args = parser.parse_args()

    print(f"Starting LoRA pretraining with model: {args.model_id}")
    print(f"Output directory: {args.output_dir}")
    print(f"BigQuery table: {args.bigquery_table_id}")

    # 1. Load data from BigQuery
    print("Loading data from BigQuery...")
    try:
        query = f"SELECT {args.narrative_column} FROM `{args.project_id}.{args.bigquery_table_id}` WHERE {args.narrative_column} IS NOT NULL AND LENGTH({args.narrative_column}) > 0"
        df = pandas_gbq.read_gbq(query, project_id=args.project_id)
        dataset = Dataset.from_pandas(df)
        print(f"Loaded {len(dataset)} examples from BigQuery.")
    except Exception as e:
        print(f"Error loading data from BigQuery: {e}")
        return

    # 2. Load pre-trained model and tokenizer
    print(f"Loading model and tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # Ensure tokenizer has a pad token for batching
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # or use tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)

    # Resize model embeddings to match new tokenizer vocabulary size if pad token was added
    if tokenizer.pad_token_id is not None and model.get_output_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    # 3. Configure LoRA
    print("Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r=8,  # LoRA attention dimension
        lora_alpha=16, # Alpha parameter for LoRA scaling
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"], # Modules to apply LoRA to
        lora_dropout=0.05, # Dropout probability for LoRA layers
        bias="none", # Bias type for LoRA layers
        task_type="CAUSAL_LM", # Task type for causal language modeling
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Prepare dataset for training
    print("Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(examples[args.narrative_column], truncation=True, max_length=args.max_seq_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=[args.narrative_column])

    # For causal language modeling, labels are usually the input IDs shifted
    # DataCollatorForLanguageModeling handles this automatically
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. Define training arguments and Trainer
    print("Initializing Trainer...")
    training_args = TrainingArguments(
        output_dir="./lora_checkpoints",  # Temporary local directory for checkpoints
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=False, # Gemma models often prefer bfloat16, or disable fp16 if not available
        bf16=True, # Use bfloat16 for Gemma
        save_strategy="epoch",
        logging_dir="./lora_logs",
        logging_steps=10,
        gradient_accumulation_steps=4, # Adjust based on GPU memory
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 6. Train the model
    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # 7. Save LoRA adapters to GCS
    # Create a local directory for saving adapters before uploading
    local_save_path = "./final_lora_adapters"
    os.makedirs(local_save_path, exist_ok=True)

    # Save only the LoRA adapters
    model.save_pretrained(local_save_path)
    print(f"LoRA adapters saved locally to: {local_save_path}")

    # Upload to GCS
    gcs_output_path = args.output_dir.rstrip('/') # Ensure no trailing slash
    bucket_name = gcs_output_path.split('/')[2] # Extract bucket name from gs://bucket/path
    bucket_path = '/'.join(gcs_output_path.split('/')[3:]) # Extract path within bucket

    # Use gsutil for directory upload
    upload_command = f"gsutil -m cp -r {local_save_path} gs://{bucket_name}/{bucket_path}/"
    print(f"Uploading adapters to GCS: {upload_command}")
    os.system(upload_command)
    print(f"LoRA adapters uploaded to {args.output_dir}/")

if __name__ == "__main__":
    main()
