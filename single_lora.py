import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb
import re
from collections import defaultdict


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
        max_length=1024,
        max_per_assay: int | None = None,
        min_score_margin: float = 0.2
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_score_margin = min_score_margin

        dms_dir = Path(dms_dir)
        csv_files = sorted(dms_dir.glob("*.csv"))

        # get all csv files in the directory except the one named DMS_substitutions.csv
        csv_files = [f for f in csv_files if f.name != "DMS_substitutions.csv"]
        if not csv_files:
            raise ValueError(f"No CSV files found in {dms_dir}")

        self.seqs: list[str] = []  # mutated sequences
        self.labels: list[float] = []  # normalized DMS scores
        self.task_ids: list[int] = []  # which assay this example belongs to
        self.mut_pos = []
        self.assay_names = []

        # To facilitate fast sampling of partners from the same assay
        self.indices_by_task = defaultdict(list)

        print(f"Found {len(csv_files)} assay CSVs in {dms_dir}")
    
        global_idx_counter = 0
        task_id = 0
        for csv_path in csv_files:
            assay_name = csv_path.stem
            self.assay_names.append(assay_name)
            df = pd.read_csv(csv_path)

            seq_col = "mutated_sequence"
            label_col = "DMS_score"
            mut_col = "mutant"



            df = df.dropna(subset=[seq_col, label_col, mut_col])

            if max_per_assay is not None and len(df) > max_per_assay:
                df = df.sample(max_per_assay, random_state=42).reset_index(drop=True)

            n_rows = len(df)
            if n_rows < 2:
                print(f"WARNING: {csv_path} has {n_rows} usable rows; skipping.")
                continue

            print(f"Loading assay {csv_path.name} as task_id={task_id} with {n_rows} rows")

            scores = df[label_col].astype(float).values
            mu = scores.mean()
            sigma = scores.std() if scores.std() > 0 else 1.0

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

                # Store the index for this task
                self.indices_by_task[task_id].append(global_idx_counter)
                global_idx_counter += 1

            # after we finish this CSV
            task_id += 1


        self.num_tasks = len(self.indices_by_task)
        if self.num_tasks == 0:
            raise ValueError("No assays loaded; please check CSV contents.")

        print(f"Total examples: {len(self.seqs)} across {self.num_tasks} assays.")

    def __len__(self):
        return len(self.seqs)

    def _get_tokenized_item(self, idx):
            """Helper to get a single tokenized object"""
            enc = self.tokenizer(
                self.seqs[idx],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "mut_pos": torch.tensor(self.mut_pos[idx], dtype=torch.long),
                "score": self.labels[idx] # Keep raw score for comparison
            }

    def __getitem__(self, idx):
        task_id = self.task_ids[idx]
        score_a = self.labels[idx]

        candidates = self.indices_by_task[task_id]
        idx_b = idx
        for _ in range(25):  # try up to 25 times to find a valid partner
            rand_idx = np.random.choice(candidates)
            if rand_idx == idx : 
                continue
            score_b = self.labels[rand_idx]
            if abs(score_a - score_b) >= self.min_score_margin:
                idx_b = rand_idx
                break

        # if we failed to find a margin pair, just take the last random one
        if idx_b == idx:
            idx_b = rand_idx

        
        # Determine winner (chosen) vs loser (rejected)
        score_b = self.labels[idx_b]
        if score_a >= score_b:
            chosen_idx, rejected_idx = idx, idx_b
        else:
            chosen_idx, rejected_idx = idx_b, idx
        
        # tokenize both
        item_chosen = self._get_tokenized_item(chosen_idx)
        item_rejected = self._get_tokenized_item(rejected_idx)

        score_chosen = self.labels[chosen_idx]
        score_rejected = self.labels[rejected_idx]


        # return flast dictionary with prefixes
        return {
            "chosen_input_ids": item_chosen["input_ids"],
            "chosen_attention_mask": item_chosen["attention_mask"],
            "chosen_mut_pos": item_chosen["mut_pos"],
            
            "rejected_input_ids": item_rejected["input_ids"],
            "rejected_attention_mask": item_rejected["attention_mask"],
            "rejected_mut_pos": item_rejected["mut_pos"],

            "task_id": torch.tensor(task_id, dtype=torch.long),
            "labels": torch.tensor([score_chosen, score_rejected], dtype=torch.float32)
        }
# ============================================================
# Data Collator
# ============================================================

def pairwise_collate_fn(batch):
    batch_data = {}
    
    # Keys that we expect in the batch
    keys = [
        "chosen_input_ids", "chosen_attention_mask", "chosen_mut_pos",
        "rejected_input_ids", "rejected_attention_mask", "rejected_mut_pos",
        "task_id", "labels"
    ]
    
    for key in keys:
        # Stack tensors
        batch_data[key] = torch.stack([item[key] for item in batch])
        
    return batch_data

# ============================================================
# Reward Model Layer
# ============================================================

class ProteinRewardModel(nn.Module):
    """
    A Reward Model (or Fitness Model) that predicts a scalar score 
    representing the 'fitness' or 'preference' of a protein sequence.
    
    Trained using Pairwise Preference Loss (Bradley-Terry).
    """
    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.score_head = nn.Linear(hidden_size, 1)
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Forward the command to enable gradient checkpointing to the base model.
        """
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        
    def forward(
        self, 
        chosen_input_ids=None, 
        chosen_attention_mask=None, 
        chosen_mut_pos=None, 
        rejected_input_ids=None, 
        rejected_attention_mask=None, 
        rejected_mut_pos=None, 
        input_ids=None, 
        attention_mask=None, 
        mut_pos=None,
        labels=None,
        **kwargs
    ):

        # ======================================================
        # MODE 1: Pairwise Training (Bradley-Terry Loss)
        # ======================================================
        if chosen_input_ids is not None and rejected_input_ids is not None:

            input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
            attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
            mut_pos = torch.cat([chosen_mut_pos, rejected_mut_pos], dim=0)

            # Base Model Forward
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )
            # Extract hidden states (last layer)
            last_hidden = outputs.hidden_states[-1]
            
            # 3. Extract embeddings at mutation positions
            batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
            target_embeddings = last_hidden[batch_indices, mut_pos]  # [B, H]
            target_embeddings = target_embeddings.to(self.score_head.weight.dtype)  # ensure dtype match

            # Predict Rewards (Fitness Scores)
            all_rewards = self.score_head(target_embeddings).squeeze(-1)  # [B]
            
            # Split and compute loss
            B = chosen_input_ids.size(0)
            chosen_rewards = all_rewards[:B]
            rejected_rewards = all_rewards[B:]

            # Loss : Maximize log(sigmoid(chosen - rejected ))
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

            return {"loss": loss, "chosen_rewards": chosen_rewards, "rejected_rewards": rejected_rewards}

        # ======================================================
        # MODE 2: Single Inference (Scoring)
        # ======================================================
        else:
            # Standard path for validating Spearman correlation or inference
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )
            hidden_states = outputs.hidden_states[-1]
            
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            regression_inputs = hidden_states[batch_indices, mut_pos]
            regression_inputs = regression_inputs.to(self.score_head.weight.dtype)

            scores = self.score_head(regression_inputs).squeeze(-1)

            return {
                'loss': None,
                'logits': scores # These are your predicted fitness values
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
# Metrics (Spearman + MSE)
# ============================================================

def compute_metrics(eval_pred):
    # Unpack Predictions
    # The Trainer passes the output of forward() (minus 'loss') as a tuple.
    # index 0: chosen_rewards [Batch_Size, 1]
    # index 1: rejected_rewards [Batch_Size, 1]
    chosen_preds, rejected_preds = eval_pred.predictions
    
    # Unpack Labels
    # label_ids shape is [Batch_Size, 2] because we stacked [score_chosen, score_rejected]
    labels = eval_pred.label_ids
    chosen_labels = labels[:, 0]
    rejected_labels = labels[:, 1]
    
    # Flatten everything to 1D arrays
    # We concatenate chosen and rejected to create one giant list of "All Predictions" vs "All Truths"
    all_preds = np.concatenate([chosen_preds.flatten(), rejected_preds.flatten()])
    all_labels = np.concatenate([chosen_labels.flatten(), rejected_labels.flatten()])
    
    # Calculate Spearman
    spearman, _ = spearmanr(all_labels, all_preds)
    
    # Optional: Calculate Pairwise Accuracy (How often did we guess the winner right?)
    # Since we defined chosen as the one with higher score, predictions should be chosen > rejected
    num_correct = np.sum(chosen_preds > rejected_preds)
    accuracy = num_correct / len(chosen_preds)
    
    return {
        "spearman": spearman,
        "pairwise_accuracy": accuracy
    }

# ============================================================
# Parse Data
# ============================================================
def parse_data(dms_dir, dataset):
    dms_dir = Path(dms_dir)
    
    # grab the csv file named DMS_substitutions
    reference_file = dms_dir / "DMS_substitutions.csv"
    df = pd.read_csv(reference_file)

    assay_names = dataset.assay_names
    assay_to_uniport = {}
    for assay_name in assay_names:
       assay_to_uniport[assay_name] = df.loc[df["DMS_id"] == assay_name, "UniProt_ID"].iloc[0]

    return assay_to_uniport

# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dms_dir", type=str, required=True, help="Directory with DMS CSV files")
    parser.add_argument("--model_name", type=str, default="microsoft/Dayhoff-170m-UR50-BRq", help="Pretrained model name or path")
    parser.add_argument("--output_dir", type=str, default=f"./multi_c_output", help="Output directory for model checkpoints")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--max_per_assay", type=int, default=1000, help="Maximum examples per assay")
    parser.add_argument("--num_train_epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Evaluation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Model save steps")
    parser.add_argument("--wandb_project", type=str, default="multi_assay_protein_dms", help="Weights & Biases project name")
    parser.add_argument("--task", type=str, choices=["task_a", "task_b"], required=True, help="Task type for LoRA configuration")
    args = parser.parse_args()


    print(args)

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

    assay_to_uniport = parse_data(args.dms_dir, dataset)
    # 1. Map Task IDs to UniProt
    task_to_prot = {i: assay_to_uniport[name] for i, name in enumerate(dataset.assay_names)}

    # 2. Split Unique Proteins
    unique_prots = list(set(task_to_prot.values()))
    np.random.shuffle(unique_prots)
    train_prots = set(unique_prots[:int(0.9 * len(unique_prots))])

    # 3. Identify which Task IDs belong to Train
    train_tasks = {t for t, p in task_to_prot.items() if p in train_prots}

    # 4. Create Subsets (Vectorized for speed)
    all_task_ids = np.array(dataset.task_ids)
    is_train = np.isin(all_task_ids, list(train_tasks))

    train_dataset = Subset(dataset, np.where(is_train)[0])
    eval_dataset = Subset(dataset, np.where(~is_train)[0])


    # # Split dataset into train and eval (80 percent of task_ids for training, 20 percent for eval)
    # task_ids = dataset.task_ids
    # unique_task_ids = list(set(task_ids))

    # # shuffle randomly (80-20 split)
    # np.random.seed(42)
    # np.random.shuffle(unique_task_ids)
    # split_idx = int(0.8 * len(unique_task_ids))
    # train_task_ids = set(unique_task_ids[:split_idx])
    # eval_task_ids = set(unique_task_ids[split_idx:])
    # train_indices = [i for i, t_id in enumerate(task_ids) if t_id in train_task_ids]
    # eval_indices = [i for i, t_id in enumerate(task_ids) if t_id in eval_task_ids]
    # train_dataset = Subset(dataset, train_indices)
    # eval_dataset = Subset(dataset, eval_indices)



    # Prepare LoRA adapters for each task
    lora_config = get_lora_config(args.task)
    base_model = get_peft_model(base_model, lora_config)

    # Re-wrap with reward model after PEFT
    model = ProteinRewardModel(base_model, hidden_size)
    
    print(f"Adapter 'adapter_task_a' added.")


    # You can verify the adapters attached
    print("Active adapters:", model.base_model.active_adapters)
    print("PEFT config:", model.base_model.peft_config) # Shows configurations for all attached adapters

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_strategy="steps",     # <- correct name
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        report_to="wandb",
        fp16=True,                       # or bf16=True, but be careful with 8-bit
        remove_unused_columns=False,    # important for custom collate_fn
        dataloader_num_workers=4,        # Speed up data loading
        gradient_checkpointing=True,    # Turn on if OOM, off for speed
        label_names=["labels"],            # Explicitly tell Trainer to pass 'labels' to compute_metrics
    )





    # --- Training Phase 1: Train only adapter_task_a ---
    print("\n--- Training Adapter A ---")
    # Ensure only LoRA parameters and regression head are trainable
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        elif 'score_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=pairwise_collate_fn,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Save
    model.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
