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
            mutated_sequence, DMS_score, fold_modulo_5
      - assigns a unique task_id to each assay file
      - splits data based on fold_modulo_5 (0 = eval, 1-4 = train)
      - ensures pairs come from the same fold during training

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
        min_score_margin: float = 0.2,
        max_score_margin: float = 0.8,
        mode = "pairwise"
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_score_margin = min_score_margin
        self.max_score_margin = max_score_margin
        self.mode = mode
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
        self.fold_modulo_5 = []  # fold_modulo_5: 0 = eval, 1-4 = train

        # To facilitate fast sampling of partners from the same assay AND same fold
        self.indices_by_task = defaultdict(list)
        self.indices_by_task_and_fold = defaultdict(lambda: defaultdict(list))  # task_id -> fold -> [indices]

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
            fold_col = "fold_modulo_5"

            # Check if fold_modulo_5 column exists
            if fold_col not in df.columns:
                raise ValueError(f"Column '{fold_col}' not found in {csv_path}. Required for gym split.")

            df = df.dropna(subset=[seq_col, label_col, mut_col, fold_col])

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
                # check for multiple mutations (e.g. "A123C:G124D") and return a list of integers for mutation positions
                pos = re.findall(r'\d+', mut_str)
                # return [int(p) - 1 for p in pos]  # 0-based
                return int(pos[0]) - 1  # only first mutation position, 0-based

            for _, row in df.iterrows():
                seq = str(row[seq_col])
                raw_score = float(row[label_col])
                mut_str = row[mut_col]
                pos = parse_pos(mut_str)
                fold = int(row[fold_col])  # 0 = eval, 1-4 = train

                # --- SKIP if out of range ---
                # if any(p < 0 or p >= len(seq) or p >= self.max_length for p in pos):
                if pos < 0 or pos >= len(seq) or pos >= self.max_length:
                    continue
                # ----------------------------

                label = (raw_score - mu) / sigma  # normalize

                self.seqs.append(seq)
                self.labels.append(label)
                self.task_ids.append(task_id)
                self.mut_pos.append(pos)
                self.fold_modulo_5.append(fold)

                # Store the index for this task
                self.indices_by_task[task_id].append(global_idx_counter)
                # Store the index for this task AND fold (for efficient same-fold pairing)
                self.indices_by_task_and_fold[task_id][fold].append(global_idx_counter)
                global_idx_counter += 1

            # after we finish this CSV
            task_id += 1


        self.num_tasks = len(self.indices_by_task)
        if self.num_tasks == 0:
            raise ValueError("No assays loaded; please check CSV contents.")

        print(f"Total examples: {len(self.seqs)} across {self.num_tasks} assays.")
        print(f"Train examples (fold 1-4): {sum(1 for f in self.fold_modulo_5 if f in [1, 2, 3, 4])}")
        print(f"Eval examples (fold 0): {sum(1 for f in self.fold_modulo_5 if f == 0)}")

    def _encode(self, idx):
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
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "task_id": torch.tensor(self.task_ids[idx], dtype=torch.long),
        }
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):


        # ---------- pointwise inference ----------
        if self.mode == "pointwise":
            item = self._encode(idx)
            # Trainer expects "labels"
            return {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
                "mut_pos": item["mut_pos"],
                "task_id": item["task_id"],
                "labels": item["label"],
            }
        

        # ---------- pairwise training ----------
        task_id = self.task_ids[idx]
        score_a = self.labels[idx]
        fold_a = self.fold_modulo_5[idx]

        candidates = self.indices_by_task_and_fold[task_id][fold_a]
        if len(candidates) < 2:
            candidates = self.indices_by_task[task_id]

        idx_b = idx
        rand_idx = idx
        for _ in range(10):
            rand_idx = int(np.random.choice(candidates))
            if rand_idx == idx:
                continue
            score_b = self.labels[rand_idx]
            if self.min_score_margin <= abs(score_a - score_b) <= self.max_score_margin:
                idx_b = rand_idx
                break
        if idx_b == idx:
            idx_b = rand_idx

        score_b = self.labels[idx_b]
        chosen_idx, rejected_idx = (idx, idx_b) if score_a >= score_b else (idx_b, idx)

        chosen = self._encode(chosen_idx)
        rejected = self._encode(rejected_idx)

        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_mut_pos": chosen["mut_pos"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_mut_pos": rejected["mut_pos"],
            "task_id": chosen["task_id"],
            # optional: keep labels around (not used by loss)
            "labels": chosen["label"],
        }
# ============================================================
# Data Collator
# ============================================================

def pairwise_collate_fn(batch):
    keys = [
        "chosen_input_ids","chosen_attention_mask","chosen_mut_pos",
        "rejected_input_ids","rejected_attention_mask","rejected_mut_pos",
        "task_id","labels"
    ]
    return {k: torch.stack([x[k] for x in batch]) for k in keys}

def pointwise_collate_fn(batch):
    keys = ["input_ids","attention_mask","mut_pos","task_id","labels"]
    return {k: torch.stack([x[k] for x in batch]) for k in keys}


# ============================================================
# Create Dataset Split - Based on fold_modulo_5
# ============================================================

def create_dataset_split(dataset, dms_dir=None, split_ratio=None, seed=None):
    """
    Split dataset based on fold_modulo_5 column:
    - fold_modulo_5 == 0 → eval
    - fold_modulo_5 in [1, 2, 3, 4] → train
    
    Note: split_ratio and seed are kept for API compatibility but not used.
    """
    
    # 1. Split based on fold_modulo_5
    all_folds = np.array(dataset.fold_modulo_5)
    is_train = np.isin(all_folds, [1, 2, 3, 4])  # folds 1-4 mean train
    is_eval = (all_folds == 0)  # fold 0 means eval
    
    # 2. Create Subsets
    train_indices = np.where(is_train)[0]
    eval_indices = np.where(is_eval)[0]

    # 3. If eval set is too large, randomly sample 1024 examples
    if len(eval_indices) > 1024:
        rng = np.random.default_rng(42)
        eval_indices = rng.choice(eval_indices, size=1024, replace=False)
    
    
    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)
    
    # Print statistics
    print(f"fold_modulo_5 split statistics:")
    print(f"  Total examples: {len(dataset)}")
    print(f"  Train examples (fold 1-4): {len(train_dataset)} ({100*len(train_dataset)/len(dataset):.1f}%)")
    print(f"  Eval examples (fold 0): {len(eval_dataset)} ({100*len(eval_dataset)/len(dataset):.1f}%)")
    
    return train_dataset, eval_dataset
    
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

    
    def _score(self, input_ids, attention_mask, mut_pos, **kwargs):
        
        # Base Model Forward
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_router_logits=False,
            **kwargs
        )

        # Extract hidden states (last layer)
        last_hidden = outputs.hidden_states[-1]  # [B,L,H]

        # Extract embeddings at mutated residue positions
        b = last_hidden.size(0)
        idx = torch.arange(b, device=last_hidden.device)

        # Get embeddings at mutation positions: [B, H]q
        emb = last_hidden[idx, mut_pos]          # [B,H]
        emb = emb.to(self.score_head.weight.dtype)

        # Predict Rewards (Fitness Scores)
        return self.score_head(emb).squeeze(-1)  # [B]
        
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

        # ---------- pointwise inference (evaluation) ----------
        if input_ids is not None:
            rewards = self._score(input_ids, attention_mask, mut_pos, **kwargs)
            # Compute MSE loss for evaluation if labels are provided
            if labels is not None:
                loss = F.mse_loss(rewards, labels)
            else:
                loss = None
            return {"loss": loss, "logits": rewards}


        # ======================================================
        # Pairwise Training (Bradley-Terry Loss)
        # ======================================================
        input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
        all_mut_pos = torch.cat([chosen_mut_pos, rejected_mut_pos], dim=0)  # [B]

        all_rewards = self._score(input_ids, attention_mask, all_mut_pos, **kwargs)
        
        # Split and compute loss
        B = chosen_input_ids.size(0)
        chosen_rewards = all_rewards[:B]
        rejected_rewards = all_rewards[B:]

        # Loss : Maximize log(sigmoid(chosen - rejected ))
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        return {"loss": loss, "logits": chosen_rewards}

# ============================================================
# Lora Config for different adaptors
# ============================================================

def get_lora_config(task: str) -> LoraConfig:

    # Configuration for Task A (Attention layers)
    lora_config_task_a = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        bias="all",
        modules_to_save=["score_head"],
    )

    # Configuration for Task B (MLP layers)
    lora_config_task_b = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        lora_alpha=1,
        bias="all",
        modules_to_save=["score_head"],
    )


    if task == "Attention":
        return lora_config_task_a
    elif task == "MLP":
        return lora_config_task_b
    else: 
        raise ValueError(f"Unknown task: {task}")
        
# ============================================================
# Metrics (Spearman + MSE)
# ============================================================

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    rho, _ = spearmanr(labels, preds)
    return {"spearman": float(rho)}


# ============================================================
# Custom Trainer (if needed)
# ============================================================

class CustomRewardTrainer(Trainer):

    # This will only work for multi-GPU?
    def save_model(self, output_dir=None, _internal_call=False):
        # 1. Let the parent class handle standard saving (LoRA, Optimizer, creation of folder)
        super().save_model(output_dir, _internal_call)
        
        # 2. CRITICAL FIX: Only the Main Process (Rank 0) should save the custom head
        if self.args.process_index == 0:
            
            # Determine directory
            if output_dir is None:
                output_dir = self.args.output_dir
            
            # Safety: Ensure directory exists (just in case super() didn't create it yet)
            os.makedirs(output_dir, exist_ok=True)

            # Unwrap DDP if necessary
            model_to_save = self.model
            if hasattr(model_to_save, 'module'):
                model_to_save = model_to_save.module
            
            # Save the score head
            if hasattr(model_to_save, 'score_head'):
                save_path = os.path.join(output_dir, "score_head.pt")
                torch.save(model_to_save.score_head.state_dict(), save_path)
                # print(f"Saved score_head to {save_path}") # Optional logging

class TwoCollatorTrainer(CustomRewardTrainer):
    def __init__(self, *args, eval_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_collator = eval_collator

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.eval_collator if self.eval_collator is not None else self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )



# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dms_dir", type=str, required=True, help="Directory with DMS CSV files")
    parser.add_argument("--model_name", type=str, default="microsoft/Dayhoff-3b-GR-HM-c", help="Pretrained model name or path")
    parser.add_argument("--output_dir", type=str, default=f"./output", help="Output directory for model checkpoints")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Evaluation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Model save steps")
    parser.add_argument("--wandb_project", type=str, default="multi_assay_protein_dms", help="Weights & Biases project name")
    parser.add_argument("--task", type=str, choices=["MLP", "Attention"], required=True, help="Task type for LoRA configuration")
    args = parser.parse_args()


    print(args)

    # Get the local rank (0 to 6) from the environment variable set by accelerate
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = None

    if local_rank != -1:
        # DDP: Force model to the specific GPU for this process
        device_map = {"": local_rank}
    else:
        # Single GPU fallback
        device_map = "auto"

    print(f"Process Rank {local_rank}: Loading model...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Load base model with bfloat16 precision
    # For DDP, don't use device_map - let accelerate/Trainer handle device placement
    # Model will be loaded to CPU, then Trainer will move it to appropriate device
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map=None if local_rank != -1 else "auto",
        torch_dtype=torch.bfloat16, # explicitly load the based model in bfloat16
        output_router_logits=False,  # Disable MoE auxiliary loss to avoid tensor size mismatch
    )



    
    # Get hidden size from model config
    hidden_size = base_model.config.hidden_size if hasattr(base_model.config, 'hidden_size') else base_model.config.d_model

    # Load dataset
    dataset_train = DMSMultiAssayDataset(
        dms_dir=args.dms_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mode="pairwise",
    )

    dataset_eval = DMSMultiAssayDataset(
        dms_dir=args.dms_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mode="pointwise",
    )

    train_dataset, _ = create_dataset_split(dataset_train)
    _, eval_dataset  = create_dataset_split(dataset_eval)

    # Prepare LoRA adapters for each task
    lora_config = get_lora_config(args.task)
    base_model = get_peft_model(base_model, lora_config)

    # Re-wrap with reward model after PEFT
    model = ProteinRewardModel(base_model, hidden_size)

    model.base_model.enable_input_require_grads()

    

    print(f"Adapter {args.task} added.")


    # You can verify the adapters attached
    print("Active adapters:", model.base_model.active_adapters)
    print("PEFT config:", model.base_model.peft_config) # Shows configurations for all attached adapters



    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        report_to="wandb",
        remove_unused_columns=False,  
        gradient_checkpointing=False,   # Disable to avoid DDP parameter reuse issues with LoRA
        ddp_find_unused_parameters=True,  # change based on lora-config : if MLP then True else False (Attention) Because some experts (trainable params) might not be used in the forward pass if you do MLP 
        gradient_accumulation_steps=4,  
        dataloader_drop_last=True,
        dataloader_num_workers=4,  # Increase data loading parallelism
        label_names=["labels"],           # Explicitly tell Trainer to pass 'labels' to compute_metrics
        bf16=True,                   
        bf16_full_eval=True,
        eval_accumulation_steps=1
    )


    # Trainer
    trainer = TwoCollatorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=pairwise_collate_fn,
        eval_collator=pointwise_collate_fn,
        compute_metrics=compute_metrics,
    )


    # Train
    trainer.train()

    # Save
    # Only the main process (Rank 0) should write to disk
    if trainer.args.process_index == 0:
        print(f"Saving model to {args.output_dir}")
        model.save_pretrained(args.output_dir)
        
        # If you need to save the score_head separately (since it's custom):
        torch.save(model.score_head.state_dict(), Path(args.output_dir) / "score_head.pt")


if __name__ == "__main__":
    main()

