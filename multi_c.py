import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb


from torch.utils.data import Dataset, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model
from scipy.stats import spearmanr
from transformers import BitsAndBytesConfig


# ============================================================
# Multi-assay Dataset
# ============================================================

class DMSMultiAssayDataset(Dataset):
    """
    Multi-assay dataset that:
      - scans a directory for CSV files (one per assay)
      - each CSV must have at least:
            mutated_sequence, DMS_score
        (if your raw ProteinGym CSVs use different names, adapt here)
      - assigns a unique task_id to each assay file
      - optionally samples up to max_per_assay rows from each assay

    Fields returned per item:
      - input_ids, attention_mask
      - labels (float)
      - task_ids (long)   <-- our "metadata" for the router
    """

    def __init__(
        self,
        dms_dir,
        tokenizer,
        max_length=2048,
        max_per_assay: int | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        dms_dir = Path(dms_dir)
        
        csv_files = sorted(dms_dir.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {dms_dir}")

        self.seqs: list[str] = []  # mutated sequences
        self.labels: list[float] = []  # normalized DMS scores
        self.task_ids: list[int] = []  # which assay this example belongs to
        self.assay_names: list[str] = []  # index -> file stem
        self.task_mean = {}
        self.task_std = {}
        self.mut_pos = []

        print(f"Found {len(csv_files)} assay CSVs in {dms_dir}")

        task_id = 0
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)

            seq_col = "mutated_sequence"
            label_col = "DMS_score"
            mut_col = "mutant"


            df = df.dropna(subset=[seq_col, label_col, mut_col])

            if max_per_assay is not None and len(df) > max_per_assay:
                df = df.sample(max_per_assay, random_state=42).reset_index(drop=True)

            n_rows = len(df)
            if n_rows == 0:
                print(f"WARNING: {csv_path} has 0 usable rows; skipping.")
                continue

            print(f"Loading assay {csv_path.name} as task_id={task_id} with {n_rows} rows")

            self.assay_names.append(csv_path.stem)
            scores = df[label_col].astype(float).values
            mu = scores.mean()
            sigma = scores.std() if scores.std() > 0 else 1.0
            self.task_mean[task_id] = mu
            self.task_std[task_id] = sigma


            import re

            def parse_pos(mut_str):
                # extract all digits in the mutation, e.g. "A123C" → "123"
                pos = re.findall(r'\d+', mut_str)
                if len(pos) == 0:
                    raise ValueError(f"Cannot find position in mutation {mut_str}")
                return int(pos[0]) - 1  # 0-based

            for _, row in df.iterrows():
                seq = str(row[seq_col])
                raw_score = float(row[label_col])
                mut_str = row[mut_col]
                pos = parse_pos(mut_str)

                # --- SKIP if out of range ---
                if pos < 0 or pos >= len(seq) or pos >= self.max_length:
                    continue
                # ----------------------------

                label = (raw_score - mu) / sigma  # normalize

                self.seqs.append(seq)
                self.labels.append(label)
                self.task_ids.append(task_id)
                self.mut_pos.append(pos)

            # after we finish this CSV
            task_id += 1


        self.num_tasks = len(self.assay_names)
        if self.num_tasks == 0:
            raise ValueError("No assays loaded; please check CSV contents.")

        print(f"Total examples: {len(self.seqs)} across {self.num_tasks} assays.")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]
        task_id = self.task_ids[idx]
        mut_pos = self.mut_pos[idx]

        enc = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float32)
        item["task_ids"] = torch.tensor(task_id, dtype=torch.long)
        item["mut_pos"] = torch.tensor(mut_pos, dtype=torch.long)

        return item

# ============================================================
# Data Collator
# ============================================================

def collate_fn(batch):
    """Custom collator that handles mut_pos and other fields."""
    collated = {}
    for key in batch[0].keys():
        if key in ['input_ids', 'attention_mask']:
            collated[key] = torch.stack([item[key] for item in batch])
        elif key in ['labels', 'task_ids', 'mut_pos']:
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]
    return collated

# ============================================================
# Regression Head Model
# ============================================================

class RegressionHeadModel(nn.Module):
    """
    Wrapper model that adds a regression head on top of a base language model.
    Uses the hidden state at the mutation position to predict DMS scores.
    """
    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.regression_head = nn.Linear(hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None, mut_pos=None, labels=None, **kwargs):
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        # Extract hidden states (last layer)
        hidden_states = outputs.hidden_states[-1]
        
        # Get hidden state at mutation position for each sample
        # mut_pos: [B], hidden_states: [B, seq_len, H]
        batch_size = hidden_states.size(0)
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        regression_inputs = hidden_states[batch_indices, mut_pos]  # [B, H]
        regression_output = self.regression_head(regression_inputs).squeeze(-1)  # [B]

        
        # Compute MSE loss if labels provided
        loss = None
        if labels is not None:
            loss = F.mse_loss(regression_output, labels)
        
        return {
            'loss': loss,
            'logits': regression_output,
        }


# ============================================================
# Lora Config for different adaptors
# ============================================================

def get_lora_config(task: str) -> LoraConfig:

    lora_config_task_a = LoraConfig(
        task_type=TaskType.CAUSAL_LM,   # this is still correct for JambaForCausalLM
        r=8,
        lora_alpha=32,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # Configuration for Task B (e.g., Sentiment Analysis)
    lora_config_task_b = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4, # Lower rank for Task B
        lora_alpha=8,
        # lora_dropout=0.1,
        target_modules=["c_proj"], # Example: Target different layers for Task B
        bias="none",
    )


    if task == "task_a":
        return lora_config_task_a
    elif task == "task_b":
        return lora_config_task_b
    else:
        raise ValueError(f"Unknown task: {task}")



# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dms_dir", type=str, required=True, help="Directory with DMS CSV files")
    parser.add_argument("--model_name", type=str, default="microsoft/Dayhoff-3b-GR-HM-c", help="Pretrained model name or path")
    parser.add_argument("--output_dir", type=str, default=f"./multi_c_output", help="Output directory for model checkpoints")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--max_per_assay", type=int, default=1000, help="Maximum examples per assay")
    parser.add_argument("--num_train_epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Model save steps")
    parser.add_argument("--wandb_project", type=str, default="multi_assay_protein_dms", help="Weights & Biases project name")
    parser.add_argument("--task", type=str, choices=["task_a", "task_b"], required=True, help="Task type for LoRA configuration")
    args = parser.parse_args()


    print(args)
    # Initialize Weights & Biases
    # wandb.init(project=args.wandb_project)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,             # 8-bit weights
        llm_int8_threshold=6.0,        # default is fine
        llm_int8_has_fp16_weight=False # usually False for full 8-bit
    )
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto"   # let HF place it on GPUs
    )

    
    # Get hidden size from model config
    hidden_size = base_model.config.hidden_size if hasattr(base_model.config, 'hidden_size') else base_model.config.d_model

    # Load dataset
    dataset = DMSMultiAssayDataset(
        dms_dir=args.dms_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_per_assay=args.max_per_assay,
    )


    # Split dataset into train and eval (80 percent of task_ids for training, 20 percent for eval)
    task_ids = dataset.task_ids
    unique_task_ids = list(set(task_ids))

    # shuffle randomly (80-20 split)
    np.random.seed(42)
    np.random.shuffle(unique_task_ids)
    split_idx = int(0.8 * len(unique_task_ids))
    train_task_ids = set(unique_task_ids[:split_idx])
    eval_task_ids = set(unique_task_ids[split_idx:])
    train_indices = [i for i, t_id in enumerate(task_ids) if t_id in train_task_ids]
    eval_indices = [i for i, t_id in enumerate(task_ids) if t_id in eval_task_ids]
    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)



    # Prepare LoRA adapters for each task
    lora_config = get_lora_config(args.task)
    base_model = get_peft_model(base_model, lora_config)
    
    # Re-wrap with regression head after PEFT
    model = RegressionHeadModel(base_model, hidden_size)
    
    print(f"Adapter 'adapter_task_a' added.")


    # You can verify the adapters attached
    print("Active adapters:", model.base_model.active_adapters)
    print("PEFT config:", model.base_model.peft_config) # Shows configurations for all attached adapters

    # Define base training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to="wandb",   # or "none"
        fp16=True,           # or bf16=True on A100/H100
    )




    # --- Training Phase 1: Train only adapter_task_a ---
    print("\n--- Training Adapter A ---")
    # Ensure only LoRA parameters and regression head are trainable
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        elif 'regression_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    # Train
    trainer.train()

    # Save
    model.base_model.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
