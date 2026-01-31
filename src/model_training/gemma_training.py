
import sys
import os
import gc
import torch

from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Add the 'src' directory to the system path to allow for absolute imports.
# This is necessary so that this script can find the 'utils' module.
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

flash_attn_available = False

def setup_model_training_environment(config):
    print(f"Setting up model training environment for {config.config_loader.get(config.MODEL_ID)} ")
    if torch.cuda.is_available():
        try:
            # Import the specific functions you need
            import flash_attn
            flash_attn_available = True
            print("FlashAttention successfully imported.")
            # Force the system to see the CUDA libraries
            os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

        except ImportError:
            print(" CUDA is available, but 'flash-attn' package is not installed.")
    else:
        print("CUDA driver not found. Falling back to standard attention.")

    # bitsandbytes which version to simulate
    os.environ['FORCE_BITSANDBYTES_LOAD'] = '1'

    #Memory Management Environment Variable
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Force fragmentation to be handled aggressively
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")      # For your M4 MacBook
    elif torch.cuda.is_available():
        device = torch.device("cuda")     # For your G4 / A100 server
    else:
        device = torch.device("cpu")      # Fallback

    print(f"Using device: {device}")
    return device


def get_attention_implementation():
    if flash_attn_available:
        return "flash_attention_2"
    return "sdpa" # Default for PyTorch 2.0+ or CPU/MPS

def get_tokeniser(cl):
    MODEL_ID = cl.config_loader.get(cl.MODEL_ID)
    HF_TOKEN = cl.config_loader.get(cl.HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_model(cl):
    MODEL_ID = cl.config_loader.get(cl.MODEL_ID)
    HF_TOKEN = cl.config_loader.get(cl.HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        #device_map="auto",
        attn_implementation=get_attention_implementation(), # Optimized for H100
        token=HF_TOKEN
    )
    model.to(get_device())
    return model

def load_and_clean_data(file_path):
    iter_csv = pd.read_csv(
        file_path,
        usecols=["Consumer complaint narrative"],
        chunksize=20000,
        encoding='utf-8'
    )
    df = pd.concat([chunk.dropna(subset=["Consumer complaint narrative"]) for chunk in iter_csv])
    df = df.rename(columns={"Consumer complaint narrative": "text"})
    return Dataset.from_pandas(df)

def calculate_perplexity(model, tokenizer, texts):
    model.eval()
    nlls = []

    # Use torch.inference_mode() for faster, more memory-efficient evaluation
    with torch.inference_mode():
        for text in tqdm(texts):
            # 1. Tokenize and immediately move ALL resulting tensors to GPU
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(model.device)

            # 2. Extract input_ids to use as labels (standard for causal PPL)
            input_ids = inputs["input_ids"]

            # 3. Calculate loss
            outputs = model(**inputs, labels=input_ids)

            # 4. Extract scalar loss and store
            nlls.append(outputs.loss)

    # Calculate final PPL: exp(mean(negative log likelihoods))
    return torch.exp(torch.stack(nlls).mean())


def configure_lora(model):
    # 1. Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()

    # 1. Force the model to the H100's native language (BF16)
    model.to(torch.bfloat16)
    model.enable_input_require_grads()
    #model.gradient_checkpointing_enable()
    #model.is_parallelizable = True
    #model.model_parallel = True

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model

def get_training(model, tokenizer, tokenized_dataset):
    is_cuda = torch.cuda.is_available()
    # TF32 is only for Ampere (A100) or newer
    is_ampere = is_cuda and torch.cuda.get_device_capability()[0] >= 8
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.to(torch.bfloat16)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 2. Configure Training for a "Testing Pass"
    training_args = TrainingArguments(
        output_dir="./gemma-fin-test",

        # Epoch Configuration
        #num_train_epochs=1,              # We only need 1 pass for adaptation
        #max_steps=100,


        # Total steps to run for this test

        num_train_epochs=1,          # Reduced from 3 to 1 for better generalizability
        max_steps=-1,                # Let it run the full epoch
        logging_steps=10,            # See updates every ~2.5 minutes
        save_strategy="steps",
        save_steps=100,

        # A2-HighGPU-4G Memory Optimization
        per_device_train_batch_size=8,   # Small batch to prevent OOM
        gradient_accumulation_steps=8,   # 2 * 8 * 4 GPUs = 64 effective batch
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Optimizer & Speed
        # optim="paged_adamw_8bit",
        optim="paged_adamw_8bit" if is_cuda else "adamw_torch", # 8bit optim requires CUDA
        #bf16=True,
        #tf32=True,
        bf16=is_cuda,             # Only True if on NVIDIA
        tf32=is_ampere,           # Only True if on A100/H100
        fp16=not is_cuda,


        learning_rate=2e-5,              # Slightly higher for 10k samples
        weight_decay=0.01,

        # Logging (How you see progress)
        #logging_steps=5,                 # Show loss every 5 steps
        logging_first_step=True,         # Check the very first loss value
        report_to="none",

        # Save only the best at the end
        #save_strategy="no",
    )
    # 3. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset.select(range(10000)), # Select only 10k
        data_collator=data_collator,
    )

    # 4. Run Training
    torch.cuda.empty_cache()
    print("Starting test run. Look for 'Loss' values around 4.0 - 8.0...")
    trainer.train()

import gc
import torch

def clear_memory():
    # 1. Clear Python's garbage collector
    gc.collect()

    # 2. Clear MPS memory if on Mac
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        # Note: torch.mps does not have 'reset_peak_memory_stats'
        print("MPS cache cleared")

    # 3. Clear CUDA memory if on G4/A100 server
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("CUDA cache cleared")