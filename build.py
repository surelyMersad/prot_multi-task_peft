import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import spearmanr
from safetensors.torch import load_file

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import get_peft_model

# Import classes and functions from train_esmc.py
from train_esmc import (
    DMSMultiAssayDataset,
    ProteinRewardModel,
    get_lora_config,
    create_dataset_split,
    pairwise_collate_fn,
    pointwise_collate_fn,
    compute_metrics,
    TwoCollatorTrainer,
)


def run_inference(
    model: nn.Module,
    dataset: DMSMultiAssayDataset,
    batch_size: int = 32,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Run inference on dataset and compute metrics."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pointwise_collate_fn,
    )

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mut_pos = batch["mut_pos"].to(device)
            labels = batch["labels"].cpu().numpy()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mut_pos=mut_pos,
            )
            rewards = outputs["logits"].cpu().numpy()

            all_preds.append(rewards)
            all_labels.append(labels)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    rho, _ = spearmanr(all_labels, all_preds)
    mse = float(((all_preds - all_labels) ** 2).mean())

    print(f"Spearman: {rho:.4f}, MSE: {mse:.4f}")
    return all_preds, all_labels, rho, mse


def main():
    parser = argparse.ArgumentParser(
        description="Load checkpoint and run inference or continue training"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True, help="Directory containing checkpoint files",
    )
    parser.add_argument(
        "--dms_dir", type=str, required=True, help="Directory with DMS CSV files",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Dayhoff-3b-GR-HM-c",
        help="Pretrained model name or path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["inference", "train"],
        required=True,
        help="Mode: inference or train",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Evaluation batch size per device",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Number of training epochs (only for train mode)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate (only for train mode)",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Logging steps (only for train mode)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Model save steps (only for train mode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for continued training (only for train mode)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["MLP", "Attention"],
        required=True,
        help="Task type for LoRA configuration",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Get local rank for DDP
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = None if local_rank != -1 else "auto"

    print(f"Loading model from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        output_router_logits=False,
    )

    hidden_size = (
        base_model.config.hidden_size
        if hasattr(base_model.config, "hidden_size")
        else base_model.config.d_model
    )

    # Load LoRA config and apply to base model
    lora_config = get_lora_config(args.task)
    base_model = get_peft_model(base_model, lora_config)

    # Wrap with reward model
    model = ProteinRewardModel(base_model, hidden_size)
    model.base_model.enable_input_require_grads()

    # Load checkpoint weights
    print(f"Loading checkpoint from {checkpoint_dir}...")
    model_path = checkpoint_dir / "model.safetensors"
    if model_path.exists():
        state_dict = load_file(str(model_path))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: Missing keys: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys: {unexpected}")
    else:
        raise ValueError(f"Model checkpoint not found: {model_path}")

    # Load score_head separately
    score_head_path = checkpoint_dir / "score_head.pt"
    if score_head_path.exists():
        model.score_head.load_state_dict(
            torch.load(str(score_head_path), map_location="cpu")
        )
    else:
        raise ValueError(f"Score head checkpoint not found: {score_head_path}")

    # Load datasets
    print("Loading datasets...")
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
    _, eval_dataset = create_dataset_split(dataset_eval)

    if args.mode == "inference":
        print("\n" + "=" * 50)
        print("Running inference on eval dataset...")
        print("=" * 50)
        # Ensure model is on the correct device
        if device_map is None:
            device = torch.device(f"cuda:{local_rank}" if local_rank != -1 else "cuda:0")
            if torch.cuda.is_available():
                model = model.to(device)
        device = next(model.parameters()).device
        run_inference(
            model,
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            device=device,
        )

    elif args.mode == "train":
        if args.output_dir is None:
            args.output_dir = str(checkpoint_dir.parent / "continued_training")

        print("\n" + "=" * 50)
        print("Continuing training...")
        print("=" * 50)

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
            gradient_checkpointing=False,
            ddp_find_unused_parameters=True,
            gradient_accumulation_steps=4,
            dataloader_drop_last=True,
            dataloader_num_workers=4,
            label_names=["labels"],
            bf16=True,
            bf16_full_eval=True,
            eval_accumulation_steps=1,
        )

        trainer = TwoCollatorTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=pairwise_collate_fn,
            eval_collator=pointwise_collate_fn,
            compute_metrics=compute_metrics,
        )

        # Resume training from checkpoint (loads optimizer, scheduler, trainer state)
        trainer.train(resume_from_checkpoint=str(checkpoint_dir))

        if trainer.args.process_index == 0:
            print(f"Saving model to {args.output_dir}")
            model.save_pretrained(args.output_dir)
            torch.save(
                model.score_head.state_dict(),
                Path(args.output_dir) / "score_head.pt",
            )


if __name__ == "__main__":
    main()

