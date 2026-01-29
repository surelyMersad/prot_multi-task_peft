import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
import os
from peft import LoraConfig, TaskType, get_peft_model
from scipy.stats import spearmanr


# ============================================================
# Constants
# ============================================================

START_TOKEN = "@"
STOP_TOKEN = "*"

# Data split configuration
TRAIN_FOLDS = [1, 2, 3, 4]
EVAL_FOLD = 0
MAX_EVAL_SAMPLES = 10

# Pairwise sampling configuration
SCORE_MARGIN_MIN = 0.2
SCORE_MARGIN_MAX = 0.8
MAX_SAMPLING_RETRIES = 10

# Required CSV columns
REQUIRED_COLUMNS = ["mutated_sequence", "DMS_score", "fold_modulo_5"]


# ============================================================
# Configuration
# ============================================================

@dataclass
class LoRATaskConfig:
    """LoRA configuration for different tasks."""
    r: int = 4
    lora_alpha: int = 1
    bias: str = "all"
    
    @staticmethod
    def get_config(task: Literal["Attention", "MLP"]) -> LoraConfig:
        """Get LoRA config based on task type."""
        base = LoRATaskConfig()
        
        target_modules = {
            "Attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
            "MLP": ["gate_proj", "up_proj", "down_proj"],
        }
        
        if task not in target_modules:
            raise ValueError(f"Unknown task: {task}. Choose 'Attention' or 'MLP'")
        
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=base.r,
            lora_alpha=base.lora_alpha,
            target_modules=target_modules[task],
            bias=base.bias,
        )


# ============================================================
# Dataset
# ============================================================

class DMSMultiAssayDataset(Dataset):
    """
    Multi-assay DMS dataset supporting both pairwise and pointwise modes.
    
    Loads multiple CSV files from a directory, each representing a different assay.
    Supports gym-style fold splitting for cross-validation.
    """

    def __init__(
        self,
        dms_dir: str,
        tokenizer,
        mode: Literal["pairwise", "pointwise"] = "pairwise",
        min_score_margin: float = SCORE_MARGIN_MIN,
        max_score_margin: float = SCORE_MARGIN_MAX,
    ):
        self.tokenizer = tokenizer
        self.mode = mode
        self.min_score_margin = min_score_margin
        self.max_score_margin = max_score_margin
        
        # Load all assays
        self._load_assays(Path(dms_dir))
        
        # Build sampling indices for pairwise mode
        if mode == "pairwise":
            self._build_sampling_indices()

    def _load_assays(self, dms_dir: Path):
        """Load all CSV files from directory."""
        csv_files = sorted(dms_dir.glob("*.csv"))
        csv_files = [f for f in csv_files if f.name != "DMS_substitutions.csv"]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {dms_dir}")

        self.sequences = []
        self.scores = []
        self.task_ids = []
        self.folds = []
        self.assay_names = []

        print(f"Loading {len(csv_files)} assays from {dms_dir}")

        for task_id, csv_path in enumerate(csv_files):
            self._load_single_assay(csv_path, task_id)

        self.num_tasks = len(self.assay_names)
        print(f"Loaded {len(self)} examples across {self.num_tasks} assays")
        print(f"  Train: {sum(f in TRAIN_FOLDS for f in self.folds)} examples")
        print(f"  Eval: {sum(f == EVAL_FOLD for f in self.folds)} examples")

    def _load_single_assay(self, csv_path: Path, task_id: int):
        """Load a single assay CSV file."""
        df = pd.read_csv(csv_path)
        
        # Validate columns
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path.name} missing columns: {missing}")
        
        # Clean data
        df = df.dropna(subset=REQUIRED_COLUMNS)
        
        if len(df) < 2:
            print(f"WARNING: {csv_path.name} has only {len(df)} rows, skipping")
            return

        print(f"  {csv_path.stem}: {len(df)} examples")
        self.assay_names.append(csv_path.stem)

        # Add to dataset
        for _, row in df.iterrows():
            self.sequences.append(str(row["mutated_sequence"]))
            self.scores.append(float(row["DMS_score"]))
            self.task_ids.append(task_id)
            self.folds.append(int(row["fold_modulo_5"]))

    def _build_sampling_indices(self):
        """Build indices grouped by task and fold for efficient pairwise sampling."""
        from collections import defaultdict
        
        self.task_fold_indices = defaultdict(lambda: defaultdict(list))
        
        for idx, (task_id, fold) in enumerate(zip(self.task_ids, self.folds)):
            self.task_fold_indices[task_id][fold].append(idx)

    def _sample_pair(self, idx: int) -> tuple[int, int]:
        """Sample a pair of sequences for pairwise ranking."""
        task_id = self.task_ids[idx]
        fold = self.folds[idx]
        score_a = self.scores[idx]
        
        # Get candidates from same fold, fallback to all task indices
        candidates = self.task_fold_indices[task_id].get(fold, [])
        if len(candidates) < 2:
            candidates = [i for i, t in enumerate(self.task_ids) if t == task_id]
        
        # Try to find a good margin pair
        idx_b = idx
        for _ in range(MAX_SAMPLING_RETRIES):
            candidate = int(np.random.choice(candidates))
            if candidate == idx:
                continue
            
            score_b = self.scores[candidate]
            margin = abs(score_a - score_b)
            
            if self.min_score_margin <= margin <= self.max_score_margin:
                idx_b = candidate
                break
        
        # If no good margin found, use last candidate
        if idx_b == idx and len(candidates) > 1:
            idx_b = candidate
        
        # Return (chosen, rejected) based on scores
        return (idx, idx_b) if score_a >= score_b else (idx_b, idx)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        if self.mode == "pointwise":
            return {
                "sequence": self.sequences[idx],
                "label": torch.tensor(self.scores[idx], dtype=torch.float32),
                "task_id": torch.tensor(self.task_ids[idx], dtype=torch.long),
            }
        
        # Pairwise mode
        chosen_idx, rejected_idx = self._sample_pair(idx)
        
        return {
            "chosen_sequence": self.sequences[chosen_idx],
            "rejected_sequence": self.sequences[rejected_idx],
            "task_id": torch.tensor(self.task_ids[chosen_idx], dtype=torch.long),
            "labels": torch.tensor(self.scores[chosen_idx], dtype=torch.float32),
        }


# ============================================================
# Data Collation
# ============================================================

def add_boundary_tokens(sequences: list[str], reverse: bool = False) -> list[str]:
    """Add start/stop tokens to sequences, optionally reversing them."""
    if reverse:
        return [STOP_TOKEN + seq[::-1] + START_TOKEN for seq in sequences]
    return [START_TOKEN + seq + STOP_TOKEN for seq in sequences]


def tokenize_sequences(
    sequences: list[str],
    tokenizer,
    reverse: bool = False,
) -> dict:
    """Tokenize sequences with boundary tokens and dynamic padding."""
    processed = add_boundary_tokens(sequences, reverse=reverse)
    
    return tokenizer(
        processed,
        return_tensors="pt",
        padding=True,  # Dynamic padding to longest in batch
        return_token_type_ids=False,
    )


def pairwise_collate_fn(batch: list[dict], tokenizer) -> dict:
    """Collate function for pairwise ranking with dynamic padding."""
    chosen = [item["chosen_sequence"] for item in batch]
    rejected = [item["rejected_sequence"] for item in batch]
    
    # Tokenize all variants: chosen/rejected × forward/backward
    chosen_fwd = tokenize_sequences(chosen, tokenizer, reverse=False)
    chosen_bwd = tokenize_sequences(chosen, tokenizer, reverse=True)
    rejected_fwd = tokenize_sequences(rejected, tokenizer, reverse=False)
    rejected_bwd = tokenize_sequences(rejected, tokenizer, reverse=True)
    
    return {
        "chosen_fwd_input_ids": chosen_fwd["input_ids"],
        "chosen_fwd_attention_mask": chosen_fwd["attention_mask"],
        "chosen_bwd_input_ids": chosen_bwd["input_ids"],
        "chosen_bwd_attention_mask": chosen_bwd["attention_mask"],
        "rejected_fwd_input_ids": rejected_fwd["input_ids"],
        "rejected_fwd_attention_mask": rejected_fwd["attention_mask"],
        "rejected_bwd_input_ids": rejected_bwd["input_ids"],
        "rejected_bwd_attention_mask": rejected_bwd["attention_mask"],
        "task_id": torch.stack([item["task_id"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def pointwise_collate_fn(batch: list[dict], tokenizer) -> dict:
    """Collate function for pointwise scoring with dynamic padding."""
    sequences = [item["sequence"] for item in batch]
    
    # Tokenize forward and backward
    fwd = tokenize_sequences(sequences, tokenizer, reverse=False)
    bwd = tokenize_sequences(sequences, tokenizer, reverse=True)
    
    return {
        "fwd_input_ids": fwd["input_ids"],
        "fwd_attention_mask": fwd["attention_mask"],
        "bwd_input_ids": bwd["input_ids"],
        "bwd_attention_mask": bwd["attention_mask"],
        "labels": torch.stack([item["label"] for item in batch]),
        "task_ids": torch.stack([item["task_id"] for item in batch]),  # Changed to plural to match label_names
    }


# ============================================================
# Dataset Splitting
# ============================================================

def create_fold_split(dataset: DMSMultiAssayDataset) -> tuple[Subset, Subset]:
    """
    Split dataset based on fold_modulo_5 values.
    For each assay, sample 200 sequences with fold_modulo_5=0 for eval.
    Remaining sequences go to train.
    """
    task_ids = np.array(dataset.task_ids)
    folds = np.array(dataset.folds)
    
    # Use deterministic random seed for reproducibility
    rng = np.random.default_rng(seed=42)
    
    eval_indices = []
    
    # Sample 200 sequences per assay with fold_modulo_5 == 0
    for task_id in np.unique(task_ids):
        # Find all sequences for this task with fold_modulo_5 == 0
        task_fold0_mask = (task_ids == task_id) & (folds == EVAL_FOLD)
        task_fold0_indices = np.where(task_fold0_mask)[0]
        
        if len(task_fold0_indices) == 0:
            print(f"  WARNING: Task {task_id} ({dataset.assay_names[task_id]}) has no sequences with fold_modulo_5=0")
            continue
        
        # Sample up to 200 sequences deterministically
        n_sample = min(200, len(task_fold0_indices))
        sampled_indices = rng.choice(task_fold0_indices, size=n_sample, replace=False)
        eval_indices.extend(sampled_indices.tolist())
        
        print(f"  Task {task_id} ({dataset.assay_names[task_id]}): sampled {n_sample}/{len(task_fold0_indices)} sequences with fold_modulo_5=0")
    
    eval_indices = np.array(eval_indices)
    
    # All other indices go to train
    all_indices = np.arange(len(dataset))
    train_indices = np.setdiff1d(all_indices, eval_indices)
    
    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)
    
    print(f"\nDataset split:")
    print(f"  Total: {len(dataset)}")
    print(f"  Train: {len(train_dataset)} ({100*len(train_dataset)/len(dataset):.1f}%)")
    print(f"  Eval: {len(eval_dataset)} ({100*len(eval_dataset)/len(dataset):.1f}%)")
    print(f"  Expected eval: {200 * dataset.num_tasks} sequences (200 per assay)")
    
    return train_dataset, eval_dataset


# ============================================================
# Model
# ============================================================

class ProteinNLLModel(torch.nn.Module):
    """
    Protein language model using negative log-likelihood for scoring.
    
    Uses bidirectional scoring: score = -max(fwd_nll, bwd_nll)
    This makes the model robust to sequence orientation.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for memory efficiency."""
        self.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def _compute_nll(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample negative log-likelihood.
        
        Returns average NLL per valid token for each sequence in the batch.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_router_logits=False,
        )
        
        logits = outputs.logits
        
        # Shift for autoregressive prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        
        # Mask padding tokens
        shift_labels = shift_labels.clone()
        shift_labels[shift_mask == 0] = -100
        
        # Compute cross-entropy loss per token
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        token_loss = F.cross_entropy(
            flat_logits, flat_labels, reduction="none", ignore_index=-100
        )
        token_loss = token_loss.view(shift_labels.shape)
        
        # Average over valid tokens per sequence
        per_sample_nll = (token_loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
        
        return per_sample_nll

    def _bidirectional_score(
        self,
        fwd_input_ids: torch.Tensor,
        fwd_attention_mask: torch.Tensor,
        bwd_input_ids: torch.Tensor,
        bwd_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute bidirectional scores using forward and backward NLL.
        
        Returns: (scores, fwd_nll, bwd_nll)
        """
        # Batch forward and backward for efficiency
        all_input_ids = torch.cat([fwd_input_ids, bwd_input_ids], dim=0)
        all_attention_mask = torch.cat([fwd_attention_mask, bwd_attention_mask], dim=0)
        
        all_nll = self._compute_nll(all_input_ids, all_attention_mask)
        
        # Split back into forward/backward
        batch_size = fwd_input_ids.size(0)
        fwd_nll, bwd_nll = torch.split(all_nll, batch_size, dim=0)
        
        # Score is negative of max NLL (lower NLL = better)
        scores = -torch.maximum(fwd_nll, bwd_nll)
        
        return scores, fwd_nll, bwd_nll

    def forward(
        self,
        # Pairwise mode inputs
        chosen_fwd_input_ids: Optional[torch.Tensor] = None,
        chosen_fwd_attention_mask: Optional[torch.Tensor] = None,
        chosen_bwd_input_ids: Optional[torch.Tensor] = None,
        chosen_bwd_attention_mask: Optional[torch.Tensor] = None,
        rejected_fwd_input_ids: Optional[torch.Tensor] = None,
        rejected_fwd_attention_mask: Optional[torch.Tensor] = None,
        rejected_bwd_input_ids: Optional[torch.Tensor] = None,
        rejected_bwd_attention_mask: Optional[torch.Tensor] = None,
        # Pointwise mode inputs
        fwd_input_ids: Optional[torch.Tensor] = None,
        fwd_attention_mask: Optional[torch.Tensor] = None,
        bwd_input_ids: Optional[torch.Tensor] = None,
        bwd_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # Pointwise mode (evaluation)
        if fwd_input_ids is not None:
            scores, fwd_nll, bwd_nll = self._bidirectional_score(
                fwd_input_ids, fwd_attention_mask,
                bwd_input_ids, bwd_attention_mask,
            )
            
            # Return average NLL as loss for logging
            avg_nll = (fwd_nll + bwd_nll) * 0.5
            loss = avg_nll.mean()
            
            return {"loss": loss, "logits": scores}
        
        # Pairwise ranking mode (training)
        # Batch all sequences for efficiency
        all_input_ids = torch.cat([
            chosen_fwd_input_ids, chosen_bwd_input_ids,
            rejected_fwd_input_ids, rejected_bwd_input_ids,
        ], dim=0)
        
        all_attention_mask = torch.cat([
            chosen_fwd_attention_mask, chosen_bwd_attention_mask,
            rejected_fwd_attention_mask, rejected_bwd_attention_mask,
        ], dim=0)
        
        all_nll = self._compute_nll(all_input_ids, all_attention_mask)
        
        # Split into components
        batch_size = chosen_fwd_input_ids.size(0)
        (
            chosen_fwd_nll,
            chosen_bwd_nll,
            rejected_fwd_nll,
            rejected_bwd_nll,
        ) = torch.split(all_nll, batch_size, dim=0)
        
        # Compute bidirectional scores
        chosen_scores = -torch.maximum(chosen_fwd_nll, chosen_bwd_nll)
        rejected_scores = -torch.maximum(rejected_fwd_nll, rejected_bwd_nll)
        
        # Ranking loss: maximize margin between chosen and rejected
        loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()
        
        return {"loss": loss, "logits": chosen_scores}


# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_pred) -> dict:
    """Compute Spearman correlation between predictions and labels."""

    preds = np.asarray(eval_pred.predictions).reshape(-1)
    labels, task_ids = eval_pred.label_ids
    labels = np.asarray(labels).reshape(-1)
    task_ids = np.asarray(task_ids).reshape(-1)

    rhos = []
    for tid in np.unique(task_ids):
        m = (task_ids == tid)
        rho, _ = spearmanr(labels[m], preds[m])
        if np.isfinite(rho):
            rhos.append(rho)

    avg_rho = float(np.mean(rhos)) if len(rhos) > 0 else float("nan")
    return {"spearman": avg_rho}


# ============================================================
# Custom Trainer
# ============================================================

class DualModeTrainer(Trainer):
    """Trainer that supports different collators for train/eval."""
    
    def __init__(self, *args, eval_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_collator = eval_collator

    def get_eval_dataloader(self, eval_dataset=None):
        """Use eval_collator for evaluation if provided."""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.eval_collator or self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override to ensure model is in eval mode and disable gradient computation."""
        # Ensure model is in eval mode
        self.model.eval()
        return super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Resume training protein language model from checkpoint"
    )
    
    # Data arguments
    parser.add_argument(
        "--dms_dir",
        type=str,
        required=True,
        help="Directory containing DMS CSV files",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Dayhoff-3b-GR-HM-c",
        help="Pretrained model name or path",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["MLP", "Attention"],
        required=True,
        help="LoRA task type (determines which modules to adapt)",
    )
    
    # Checkpoint argument
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., saved_output_NLL/checkpoint-34200)",
    )
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--wandb_project", type=str, default="multi_assay_protein_dms")
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="Wandb run ID to resume. If not provided, will try to extract from checkpoint or create new run.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name. If not provided, will use output_dir.",
    )
    
    args = parser.parse_args()
    print(f"\nArguments:\n{args}\n")

    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    if not checkpoint_path.is_dir():
        raise ValueError(f"Checkpoint path must be a directory: {checkpoint_path}")
    
    # Check for required checkpoint files
    required_files = ["model.safetensors", "trainer_state.json"]
    missing_files = [f for f in required_files if not (checkpoint_path / f).exists()]
    if missing_files:
        raise ValueError(f"Checkpoint missing required files: {missing_files}")
    
    print(f"Resuming from checkpoint: {checkpoint_path}")
    
    # Try to extract wandb run ID from checkpoint if not provided
    wandb_run_id = args.wandb_run_id
    if wandb_run_id is None:
        # Try to find wandb run ID from trainer_state.json
        trainer_state_path = checkpoint_path / "trainer_state.json"
        if trainer_state_path.exists():
            try:
                with open(trainer_state_path) as f:
                    trainer_state = json.load(f)
                    # Check if there's a wandb run ID stored
                    if "wandb_run_id" in trainer_state:
                        wandb_run_id = trainer_state["wandb_run_id"]
                        print(f"Found wandb run ID in checkpoint: {wandb_run_id}")
            except Exception as e:
                print(f"Could not read trainer_state.json: {e}")
    
    # Get the local rank (0 to 6) from the environment variable set by accelerate
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # Set wandb environment variables for resuming runs
    # The Trainer's wandb callback will pick these up automatically
    if wandb_run_id is not None:
        os.environ["WANDB_RUN_ID"] = wandb_run_id
        os.environ["WANDB_RESUME"] = "allow"  # Resume if exists, create new if not
        if local_rank == 0 or local_rank == -1:
            print(f"Setting wandb environment variables to resume run: {wandb_run_id}")
            print(f"  WANDB_RUN_ID={wandb_run_id}")
            print(f"  WANDB_RESUME=allow")
    else:
        if local_rank == 0 or local_rank == -1:
            print("No wandb run ID provided. Trainer will create a new wandb run.")
    device_map = None

    if local_rank != -1:
        # DDP: Force model to the specific GPU for this process
        device_map = {"": local_rank}
    else:
        # Single GPU fallback
        device_map = "auto"

    print(f"Process Rank {local_rank}: Loading model...")

    # Load tokenizer and base model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map=None if local_rank != -1 else "auto",
        torch_dtype=torch.bfloat16,
        output_router_logits=False,
    )

    # Load datasets with different modes
    print("\nLoading datasets...")
    dataset_train = DMSMultiAssayDataset(
        dms_dir=args.dms_dir,
        tokenizer=tokenizer,
        mode="pairwise",
    )

    dataset_eval = DMSMultiAssayDataset(
        dms_dir=args.dms_dir,
        tokenizer=tokenizer,
        mode="pointwise",
    )

    # Create train/eval splits
    train_dataset, _ = create_fold_split(dataset_train)
    _, eval_dataset = create_fold_split(dataset_eval)

    # Apply LoRA (same config as original training)
    print(f"\nApplying LoRA adapter for task: {args.task}")
    lora_config = LoRATaskConfig.get_config(args.task)
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # Wrap in NLL model
    model = ProteinNLLModel(base_model)
    model.base_model.enable_input_require_grads()
    
    # Ensure model is in eval mode for consistent evaluation
    model.eval()

    # Create collators (no max_length needed)
    train_collator = lambda batch: pairwise_collate_fn(batch, tokenizer)
    eval_collator = lambda batch: pointwise_collate_fn(batch, tokenizer)

    # Training arguments
    # If wandb is already initialized (resumed run), Trainer will use it
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        report_to="wandb",
        run_name=args.wandb_run_name or args.output_dir,
        bf16=True,
        bf16_full_eval=True,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        dataloader_num_workers=4,
        label_names=["labels", "task_ids"],
        eval_accumulation_steps=1,
        ddp_find_unused_parameters=True,
    )

    # Create trainer
    trainer = DualModeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_collator,
        eval_collator=eval_collator,
        compute_metrics=compute_metrics,
    )

    # Resume training from checkpoint
    # This will load model weights, optimizer state, scheduler state, and trainer state
    print(f"\nResuming training from checkpoint: {checkpoint_path}")
    trainer.train(resume_from_checkpoint=str(checkpoint_path))

    # Save final model
    if trainer.args.process_index == 0:
        print(f"\nSaving model to {args.output_dir}")
        model.base_model.save_pretrained(args.output_dir)
        print("Training complete!")


if __name__ == "__main__":
    main()

