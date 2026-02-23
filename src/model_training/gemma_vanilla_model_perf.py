import argparse
import os
import sys 
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load
from util.config_loader import config_loader

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

def get_perplexity_score(model, tokenizer, dataset, device):
    model.eval()
    
    max_length = min(tokenizer.model_max_length, 4096) 
    stride = 512

    encodings = tokenizer(
        "\n\n".join(dataset["narrative_text"]), 
        return_tensors="pt"
    )
    
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        
        # We only want to predict the "new" tokens in this window (the stride)
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # loss is calculated on tokens where label != -100
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

def get_perplexity_score1(model, tokenizer, dataset, device):
    model.eval()
    max_length = tokenizer.model_max_length
    stride = 512

    # Prepare the data for perplexity calculation
    encodings = tokenizer("\n\n".join(dataset["narrative_text"]), return_tensors="pt", truncation=True, max_length=max_length)

    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # length of target text
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()


def main(model_id, project_id, bigquery_dataset, output_bucket):
    # Load tokenizer and model
    hf_token = os.getenv('HF_TOKEN')
    print(f"Downloading tokenizer and model for {model_id}...{hf_token}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded to device: {device}")

    # Retrieve sample data from BigQuery
    print(f"Retrieving sample data from BigQuery: {project_id}.{bigquery_dataset}.complaints_features...")
    query = f"""SELECT complaint_id, narrative_text FROM `{project_id}.{bigquery_dataset}.complaints_features` LIMIT 100"""
    df = pd.read_gbq(query, project_id=project_id)
    print(f"Retrieved {len(df)} rows from BigQuery.")

    # Calculate perplexity
    print("Calculating perplexity...")
    perplexity_score = get_perplexity_score(model, tokenizer, df, device)
    print(f"Perplexity Score: {perplexity_score}")

    # Optionally save the perplexity score to a file in GCS
    # This part is illustrative, actual saving logic might vary
    output_file = f"gs://{output_bucket}/gemma_vanilla_perplexity.txt"
    with open("gemma_vanilla_perplexity.txt", "w") as f:
        f.write(f"Perplexity Score for vanilla {model_id}: {perplexity_score}\n")

    # Upload to GCS if needed
    print(f"Perplexity score saved locally. To upload to GCS, use: gsutil cp gemma_vanilla_perplexity.txt {output_file}")
    #os.system(f"gsutil cp gemma_vanilla_perplexity.txt {output_file}")

    mount_path = f"/gcs/cfpb-raw-lake-sdk"
    output_filename = "gemma_vanilla_perplexity.txt"
    full_path = os.path.join(mount_path, output_filename)

    try:
        # With gcsfuse, you treat the bucket like a local directory
        print(f"Saving perplexity score directly to gcsfuse mount: {full_path}")

        with open(full_path, "w") as f:
            f.write(f"Perplexity Score for vanilla {model_id}: {perplexity_score}\n")

        print("File successfully saved to GCS via gcsfuse.")

    except FileNotFoundError:
        print(f"Error: Mount path {mount_path} not found. Is gcsfuse running?")
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate perplexity of a vanilla Gemma model on BigQuery data.")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-4b-it", help="Hugging Face model ID.")
    
    # Prioritize environment variables, then config_loader defaults
    project_id_default = os.getenv('GCP_PROJECT_ID_CM', config_loader.get('project.id'))
    output_bucket_default = os.getenv('GCS_BUCKET_CM', config_loader.get('GCS_BUCKET'))
    bigquery_dataset_default = config_loader.get('bigquery.dataset')

    parser.add_argument("--project_id", type=str, default=project_id_default, help="GCP Project ID.")
    parser.add_argument("--bigquery_dataset", type=str, default=bigquery_dataset_default, help="BigQuery dataset name.")
    parser.add_argument("--output_bucket", type=str, default=output_bucket_default, help="GCS bucket for output.")

    args = parser.parse_args()
    main(args.model_id, args.project_id, args.bigquery_dataset, args.output_bucket)