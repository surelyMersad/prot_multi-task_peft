import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb


from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model
from scipy.stats import spearmanr


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

            # Try to be robust: look for common column names
            if "mutated_sequence" in df.columns:
                seq_col = "mutated_sequence"
            else:
                raise ValueError(
                    f"{csv_path} has no 'mutated_sequence' column."
                )

            if "DMS_score" in df.columns:
                label_col = "DMS_score"
            elif "score" in df.columns:
                label_col = "score"
            else:
                raise ValueError(
                    f"{csv_path} has no 'DMS_score' or 'score' column."
                )
            
            if "mutant" in df.columns:
                mut_col = "mutant"
            else:
                raise ValueError(
                    f"{csv_path} has no 'mutant' or 'mutations' column."
                )

            df = df.dropna(subset=[seq_col, label_col])

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
# Router + Skill Adapters + Regression Head
# ============================================================

class TaskRouter(nn.Module):
    """
    Router: maps task_ids -> mixture weights over K skill adapters.

    For now, it only uses task_ids (no explicit metadata).
    Later we can replace this with WT-sequence embeddings.
    """

    def __init__(self, num_tasks: int, num_skills: int, task_dim: int = 32):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_skills = num_skills

        self.task_emb = nn.Embedding(num_tasks, task_dim)
        self.mlp = nn.Sequential(
            nn.Linear(task_dim, task_dim),
            nn.ReLU(),
            nn.Linear(task_dim, num_skills),
        )

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        """
        task_ids: (B,) long
        returns: mixture weights of shape (B, num_skills), rows sum to 1
        """
        x = self.task_emb(task_ids)          # (B, task_dim)
        logits = self.mlp(x)                # (B, num_skills)
        weights = F.softmax(logits, dim=-1) # (B, num_skills)
        return weights


class SkillAdapter(nn.Module):
    """
    Simple skill adapter that operates on the pooled representation.

    This is a shallow adapter:
        y = x + GELU(Wx)
    Later we can move skills into the transformer layers / LoRA blocks.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        return x + self.act(self.linear(x))


class DayhoffMultiSkillRegression(nn.Module):
    """
    Base LM (Dayhoff) + shared LoRA + multi-skill adapter bank + router.

    - base_lm: AutoModelForCausalLM with PEFT LoRA applied
    - num_skills: number of skill adapters in bank
    - num_tasks: number of distinct assays / task_ids
    """

    def __init__(
        self,
        base_lm: nn.Module,
        hidden_size: int,
        num_tasks: int,
        num_skills: int = 4,
        task_dim: int = 32,
    ):
        super().__init__()
        self.base_lm = base_lm
        self.hidden_size = hidden_size
        self.num_tasks = num_tasks
        self.num_skills = num_skills

        self.router = TaskRouter(num_tasks=num_tasks, num_skills=num_skills, task_dim=task_dim)
        self.skill_adapters = nn.ModuleList(
            [SkillAdapter(hidden_size) for _ in range(num_skills)]
        )

        self.reg_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor, # shape  = (Batch size, Seq Length)
        attention_mask: torch.Tensor = None,
        task_ids: torch.Tensor = None, # which assay this example belongs to. shape = (Batch size,)
        labels: torch.Tensor = None,
        mut_pos: torch.Tensor = None,
    ):
        # 1) Base LM forward
        outputs = self.base_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]  # (B, T, D)

        # Mean pooling over non-pad tokens
        # if attention_mask is not None:
        #     mask = attention_mask.unsqueeze(-1)
        #     summed = (hidden * mask).sum(dim=1)
        #     counts = mask.sum(dim=1).clamp(min=1)
        #     pooled = summed / counts  # (B, D)
        # else:
        #     pooled = hidden[:, 0, :]  # (B, D)

        # position of the mutated residue
        batch_size = hidden.size(0)
        batch_idx = torch.arange(batch_size, device=hidden.device) # (B,)
        mutated_emb = hidden[batch_idx, mut_pos, :]  # (B, D)

        # 2) Router -> skill mixture weights
        if task_ids is None:
            batch_size = mutated_emb.size(0)
            task_ids = torch.zeros(batch_size, dtype=torch.long, device=mutated_emb.device)

        skill_weights = self.router(task_ids)  # (B, num_skills)

        # 3) Apply each skill adapter
        skill_outputs = []
        for adapter in self.skill_adapters:
            skill_outputs.append(adapter(mutated_emb))
        skill_outputs = torch.stack(skill_outputs, dim=1)  # (B, num_skills, D)

        skill_weights_expanded = skill_weights.unsqueeze(-1)  # (B, num_skills, 1)
        composed = (skill_outputs * skill_weights_expanded).sum(dim=1)  # (B, D)

        # 4) Regression head
        preds = self.reg_head(composed).squeeze(-1)  # (B,)

        # 5) Loss (MSE)
        loss = None
        if labels is not None:
            labels = labels.float()
            preds = preds.to(dtype=labels.dtype)
            loss = F.mse_loss(preds, labels)

        return {"loss": loss, "logits": preds}


# ============================================================
# Metrics (Spearman + MSE)
# ============================================================

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.squeeze()
    labels = labels.squeeze()

    mse = float(((preds - labels) ** 2).mean())

    if np.std(preds) == 0 or np.std(labels) == 0:
        rho = 0.0
    else:
        rho, _ = spearmanr(labels, preds)

    return {"spearman": rho, "mse": mse}


# ============================================================
# Main training entrypoint
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HF model id, e.g. microsoft/Dayhoff-3b-GR-HM",
    )
    parser.add_argument(
        "--dms_dir",
        type=str,
        required=True,
        help="Directory containing multiple DMS CSVs (one assay per CSV).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max token length for sequences",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA + head + router + skills",
    )
    parser.add_argument(
        "--max_per_assay",
        type=int,
        default=256,
        help="Maximum number of mutants to sample per assay (for sanity / speed).",
    )
    parser.add_argument(
        "--num_skills",
        type=int,
        default=4,
        help="Number of skill adapters in the bank.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="dayhoff_multi_skill",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional W&B run name",
    )

    args = parser.parse_args()

    # -------------------------
    # Weights & Biases setup
    # -------------------------
    os.environ["WANDB_PROJECT"] = args.wandb_project
    # Optional: group/notes/tags can also be set via env if you want
    if args.wandb_run_name is not None:
        os.environ["WANDB_NAME"] = args.wandb_run_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # Tokenizer & Base LM
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    base_lm = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    # -------------------------
    # Shared LoRA on base LM
    # -------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # adjust if needed
    )
    peft_lm = get_peft_model(base_lm, lora_config)
    peft_lm.print_trainable_parameters()

    hidden_size = peft_lm.config.hidden_size

    # -------------------------
    # Multi-assay Dataset
    # -------------------------
    full_ds = DMSMultiAssayDataset(
        dms_dir=args.dms_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_per_assay=args.max_per_assay,
    )
    num_tasks = full_ds.num_tasks
    print(f"Num tasks (assays): {num_tasks}")

    # Global random split across all examples (sanity-phase)
    n = len(full_ds)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test])

    print(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # -------------------------
    # Build model
    # -------------------------
    model = DayhoffMultiSkillRegression(
        base_lm=peft_lm,
        hidden_size=hidden_size,
        num_tasks=num_tasks,
        num_skills=args.num_skills,
        task_dim=32,
    ).to(device)

    # -------------------------
    # TrainingArguments
    # -------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,      # e.g. try 1e-4 or 5e-5
        num_train_epochs=args.epochs,

        do_eval=True,
        eval_steps=200,                         # only effective if your HF version supports step-wise eval
        save_steps=200,
        save_total_limit=1,

        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to=["wandb"],
        run_name=args.wandb_run_name,


        max_grad_norm=1.0,                      # 🔥 important for stability
        weight_decay=0.01,                      # optional but usually helpful
    )



    # -------------------------
    # Trainer
    # -------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # -------------------------
    # Final evaluation on val + test
    # -------------------------
    print("\nFinal evaluation on validation set:")
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    print(val_metrics)

    print("\nEvaluation on test set:")
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    print(test_metrics)

    # -------------------------
    # per assay denormalized spearman
    # -------------------------
        # -------------------------
    # Per-assay Spearman on test set (ProteinGym-style)
    # -------------------------
    from collections import defaultdict

    print("\nComputing per-assay Spearman on test set (denormalized)...")

    # 1) Get predictions from Trainer on test set
    pred_output = trainer.predict(test_ds)
    preds_norm = pred_output.predictions.squeeze()   # normalized preds
    labels_norm = pred_output.label_ids.squeeze()    # normalized labels

    # 2) Map Subset indices back to full_ds to recover task_ids + per-task mean/std
    test_indices = test_ds.indices                  # indices into full_ds
    task_ids = np.array([full_ds.task_ids[i] for i in test_indices])

    # 3) Denormalize per-assay
    raw_preds = []
    raw_labels = []
    for i, tid in enumerate(task_ids):
        mu = full_ds.task_mean[tid]
        sigma = full_ds.task_std[tid]
        raw_preds.append(preds_norm[i] * sigma + mu)
        raw_labels.append(labels_norm[i] * sigma + mu)

    raw_preds = np.array(raw_preds)
    raw_labels = np.array(raw_labels)

    # 4) Group by task_id and compute Spearman per assay
    per_assay_spearman = {}
    for tid in np.unique(task_ids):
        mask = (task_ids == tid)
        y_true = raw_labels[mask]
        y_pred = raw_preds[mask]

        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            rho = 0.0
        else:
            rho, _ = spearmanr(y_true, y_pred)

        assay_name = full_ds.assay_names[tid]
        per_assay_spearman[assay_name] = rho

    # 5) Aggregate
    all_rhos = np.array(list(per_assay_spearman.values()))
    mean_rho = float(all_rhos.mean())
    median_rho = float(np.median(all_rhos))

    print(f"\nPer-assay Spearman on test set:")
    print(f"  Mean Spearman over {len(per_assay_spearman)} assays:   {mean_rho:.4f}")
    print(f"  Median Spearman over {len(per_assay_spearman)} assays: {median_rho:.4f}")

    # Optional: print a few best/worst assays
    sorted_assays = sorted(per_assay_spearman.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 assays by Spearman:")
    for name, rho in sorted_assays[:10]:
        print(f"  {name:40s}  Spearman = {rho:.4f}")

    print("\nBottom 10 assays by Spearman:")
    for name, rho in sorted_assays[-10:]:
        print(f"  {name:40s}  Spearman = {rho:.4f}")


    # -------------------------
    # Sanity: inspect router weights for each task_id
    # -------------------------
    print("\nSanity: router mixture weights for each task (mean over a few samples)")
    model.eval()

    # For each task_id, grab up to 8 examples and print average mixture
    num_tasks = full_ds.num_tasks
    device = next(model.parameters()).device

    # Build indices per task_id
    indices_per_task = {tid: [] for tid in range(num_tasks)}
    for idx, task_id in enumerate(full_ds.task_ids):
        if len(indices_per_task[task_id]) < 8:  # cap at 8 for speed
            indices_per_task[task_id].append(idx)

    with torch.no_grad():
        for tid in range(num_tasks):
            idxs = indices_per_task.get(tid, [])
            if not idxs:
                continue
            # Grab those examples
            batch = [full_ds[i] for i in idxs]
            input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
            attention_mask = torch.stack([b["attention_mask"] for b in batch]).to(device)
            task_ids = torch.stack([b["task_ids"] for b in batch]).to(device)

            weights = model.router(task_ids)  # (B, num_skills)
            mean_weights = weights.mean(dim=0)  # (num_skills,)

            assay_name = full_ds.assay_names[tid] if tid < len(full_ds.assay_names) else f"task_{tid}"
            print(f"  Task {tid} ({assay_name}): mean mixture = {mean_weights.cpu().numpy()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
