---
language: en
tags:
- mtg
- draft
- behavioral-cloning
- reinforcement-learning
license: mit
datasets:
- 17lands
metrics:
- accuracy
model-index:
- name: mtg-draft-bc
  results:
  - task:
      type: classification
      name: Draft Pick Prediction
    metrics:
    - type: accuracy
      value: 0.6802
      name: Test Accuracy
    - type: accuracy
      value: 0.9438
      name: Top-3 Accuracy
---

# MTG Draft Model (Behavioral Cloning)

## Model Description

This model predicts human draft picks in Magic: The Gathering based on the current pack and player's card pool. It was trained via behavioral cloning on 17Lands data.

## Training Data

- **Source**: 17Lands Public Draft Data
- **Sets**: FDN, DSK, BLB, TLA
- **Samples**: 500,000
- **Rank Filter**: gold and above

## Performance

| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | 67.73% | 68.02% |
| Top-3 Accuracy | - | 94.38% |
| Loss | - | 0.8396 |

## Model Architecture

- **Embedding Dimension**: 128
- **Hidden Dimension**: 256
- **Parameters**: 308,994

## Training Details

- **Epochs**: 36 (early stopped)
- **Batch Size**: 256
- **Learning Rate**: 0.0001
- **Early Stopping**: 10 epochs patience

## Usage

```python
import torch
from train_draft import DraftEmbeddingModel

# Load model
checkpoint = torch.load('best.pt')
model = DraftEmbeddingModel(
    vocab_size=len(checkpoint['card_to_idx']),
    embed_dim=128,
    hidden_dim=256,
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Limitations

- Trained only on FDN, DSK, BLB, TLA
- Does not consider draft position (pick/pack number)
- Behavioral cloning ceiling: matches but cannot exceed human performance

## Training Date

2026-01-24
