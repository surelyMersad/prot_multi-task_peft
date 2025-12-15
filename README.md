# Improving Mutation Effect Prediction with LoRA-Fine-Tuned Protein Language Models

This repository explores **parameter-efficient fine-tuning of protein language models (PLMs)** to improve mutation effect prediction on the **ProteinGym substitution benchmark**. We fine-tune Microsoft’s **Dayhoff** models using **LoRA** and a **contrastive training objective**, evaluating generalization on unseen proteins.

---

## Models

Results are reported for the strongest baseline models at each scale:
- **170M-UR50-BRq**
- **3B-GR-HM-c** (trained with aligned homologs)

---

## Dataset

- **ProteinGym substitution benchmark**
- Assays are split by **UniProt ID** to ensure protein-level generalization
- **90% train / 10% test split**
- All mutations from a given protein belong to the same split

---

## Method

### Representation
For each mutation, we extract the **final hidden-layer embedding at the mutated residue position** from the PLM.

### Prediction Head
A lightweight **MLP score head** maps mutation-position embeddings to a scalar mutation-effect score.

### Training Objective
We train using a **Bradley–Terry contrastive loss**, which optimizes relative ranking of mutation effects rather than absolute regression targets.

### Parameter-Efficient Fine-Tuning
- **LoRA adapters applied to all MLP layers**
- Attention-only LoRA was evaluated but did not improve performance

---

## Evaluation

- Metric: **Spearman rank correlation**
- Computed on **held-out proteins unseen during training**
- Measures generalization across protein families rather than memorization

---

## Training & Evaluation Curves

Interactive training and evaluation metrics are available in the W&B report:

👉 **[View W&B Report]([https://wandb.ai/mhrezaei1/huggingface/reports/eval-spearman-25-12-15-02-41-55---VmlldzoxNTM3MzgwMg]))**


## Key Observations

1. **Contrastive Loss Improves Learning**  
   Using a contrastive (Bradley–Terry) loss consistently outperformed MSE regression, improving both training stability and evaluation metrics  
   ([Chen et al., 2023](https://arxiv.org/pdf/2305.03136)).

2. **MLP Layers Are Critical for LoRA Adaptation**  
   Injecting LoRA adapters only into attention layers led to weak or unstable learning. Applying LoRA to **all MLP layers** produced stable improvements, consistent with prior analyses of transformer adaptation  
   ([Thinking Machines Lab, 2023](https://thinkingmachines.ai/blog/lora/)).

3. **Local Mutation Representations Matter**  
   Feeding the **mutation-position embedding** directly into the score head improved performance. Averaging embeddings across the sequence did not yield gains, indicating that mutation effects are best captured through localized representations.

---

## Summary

This work shows that **contrastive training + MLP-focused LoRA adaptation** can significantly improve mutation effect prediction in PLMs, while preserving strong generalization to unseen proteins.
