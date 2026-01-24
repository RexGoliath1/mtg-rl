# Transfer Learning Analysis for MTG Draft Models

## Executive Summary

**Can our model architecture support transfer learning?**

| Aspect | v1 (Keyword) | v2 (Text Embedding) | Notes |
|--------|--------------|---------------------|-------|
| Card Embeddings | No | **Yes** | Text embeddings generalize to unseen cards |
| New Mechanics | No | **Yes** | Sentence transformers understand new text |
| Architecture Weights | Partial | **Yes** | Structural features transfer |
| Learned Synergies | No | Partial | Some patterns generalize |
| Training Pipeline | Yes | Yes | Same code for any set |

**2024 Research Benchmark** ([arXiv:2407.05879](https://arxiv.org/html/2407.05879v1)):
- Known cards: **68% accuracy** (matches our v1 model!)
- Unseen cards: **42.87% accuracy** (vs 0% with keyword encoding)

## Why MTG Draft is Different from Image Classification

### Image Classification Transfer Learning

```
ImageNet Pretraining → Fine-tune on Dogs vs Cats

Why it works:
- Low-level features (edges, textures) are universal
- Hierarchical representations build on each other
- All images share RGB pixel space
```

### MTG Draft "Transfer Learning"

```
FDN Training → Apply to DSK?

Why it's hard:
- Card vocabularies are largely disjoint
- Each set has unique mechanics and synergies
- "Good draft strategy" is set-specific
```

## Detailed Analysis

### 1. Card Embedding Layer

The embedding layer maps card names to vectors:

```python
self.card_embedding = nn.Embedding(vocab_size, embed_dim)
```

**Transfer potential**: LIMITED

- Cards that appear in multiple sets (basic lands, reprints) could share embeddings
- But most cards are set-specific (FDN has 286 unique cards, DSK has different ~280)
- Cross-set card overlap is typically <5%

**Solution**: Shared card encoder (see below)

### 2. Hidden Layers

The MLP layers learn:
- Which card features matter
- How to combine pack and pool information
- Scoring function for picks

**Transfer potential**: VERY LIMITED

- The hidden representations are tied to the specific card embeddings
- A "good creature" in FDN may have different characteristics than in DSK
- Synergies (card A + card B is good) are completely set-specific

### 3. Output Layer

The output is a distribution over pack cards.

**Transfer potential**: NONE

- Output dimension changes with vocabulary size
- Card indices don't correspond across sets

## Approaches That Could Work

### A. Multi-Set Training (Current Approach)

Train on all sets simultaneously:

```python
dataset = SeventeenLandsDataset(
    sets=["FDN", "DSK", "BLB", "MKM", "LCI"],
    ...
)
```

**Pros**:
- Single model handles all sets
- May learn general draft principles
- Simpler deployment

**Cons**:
- Larger vocabulary, larger model
- May dilute set-specific knowledge
- Memory constraints

### B. Text Embedding Encoder (Recommended - See CARD_ENCODING.md)

Create a card encoder that uses **sentence transformers** for card text:

```python
class HybridCardEncoder(nn.Module):
    """
    Encode cards using text embeddings + structural features.

    This approach handles:
    - New mechanics (text embedding captures semantics)
    - Parameterized abilities ("Mill 3" vs "Mill 5")
    - Complex abilities (full oracle text embedded)
    """

    def __init__(self, config):
        # Text embedding (handles new mechanics automatically)
        self.text_embedder = PretrainedTextEmbedder(text_config)
        self.text_proj = nn.Linear(384, config.d_model // 2)

        # Structural features (precise numerical data)
        self.mana_proj = nn.Linear(11, config.d_model // 4)
        self.type_proj = nn.Linear(30, config.d_model // 4)

        # Self-attention for card interactions
        self.interaction_layers = nn.ModuleList([
            CardInteractionLayer(config) for _ in range(2)
        ])

    def forward(self, card_texts, mana_features, type_features):
        # Text captures abilities/mechanics
        text_emb = self.text_proj(self.text_embedder.embed_batch(card_texts))

        # Structural features for precise stats
        mana_emb = self.mana_proj(mana_features)
        type_emb = self.type_proj(type_features)

        # Combine and model interactions
        combined = torch.cat([text_emb, mana_emb, type_emb], dim=-1)
        for layer in self.interaction_layers:
            combined = layer(combined)

        return combined
```

**Why text embeddings transfer**:
- Sentence transformers trained on massive text corpora
- New keyword "Toxic 2" → semantic meaning of "toxic" + "2" captured
- "Mill 3 cards" vs "Mill 5 cards" → different embeddings (quantity preserved)
- No vocabulary updates needed for new sets

### C. Property-Based Encoder (Current Implementation)

Our `shared_card_encoder.py` uses fixed keyword vocabulary:

```python
KEYWORDS = ["Flying", "First strike", "Trample", ...]  # 40 keywords
```

**Limitations**:
- Fails on new mechanics (Toxic, Connive, etc.)
- Loses parameterized info (Mill 3 vs Mill 5)
- Requires vocabulary updates each set

### C. Meta-Learning

Train a model that can quickly adapt to new sets:

```python
# MAML-style approach
for episode in training_episodes:
    # Sample a set
    set_data = sample_set(["FDN", "DSK", "BLB"])

    # Inner loop: adapt to this set
    adapted_model = model.clone()
    for step in range(k_adaptation_steps):
        loss = compute_loss(adapted_model, set_data.support)
        adapted_model = gradient_step(adapted_model, loss)

    # Outer loop: optimize for fast adaptation
    meta_loss = compute_loss(adapted_model, set_data.query)
    model = gradient_step(model, meta_loss)
```

**Pros**:
- Can adapt to new sets with few samples
- Learns "how to learn draft"

**Cons**:
- Complex training procedure
- May need more data overall

### D. Pre-training on Card Properties + Fine-tuning

1. Pre-train on predicting card properties from name/text
2. Use pre-trained card representations
3. Fine-tune on draft prediction

Similar to how BERT pre-trains on language before fine-tuning on tasks.

## Recommendations

### Phase 1: v1.0 (Completed)

- [x] Multi-set training on FDN, DSK, BLB, TLA
- [x] 68% test accuracy achieved
- [x] Model registered in `model_registry.json`
- [x] Auto-shutdown infrastructure for cost savings

### Phase 2: v2.0 (Next - Text Embeddings)

1. **Integrate text embeddings** from `text_embeddings.py` into encoder
2. **Pre-compute embeddings** for all Standard-legal cards (cache to disk)
3. **Train v2 model** with hybrid encoding (text + structural)
4. **Evaluate transfer**: Train on FDN/DSK/BLB, test on TLA (held out)
5. **Compare v1 vs v2** accuracy and transfer performance

### Phase 3: Production

1. **A/B test** v1 (keyword) vs v2 (text) in draft assistant
2. **Monitor new set performance** when new set releases
3. **Continual learning**: Fine-tune on new set data without forgetting

## Code Structure for Transfer Learning

```
mtg/
├── encoders/
│   ├── card_name_encoder.py      # Current: name → embedding
│   ├── shared_card_encoder.py    # Future: properties → embedding
│   └── card_text_encoder.py      # Future: text → embedding
├── models/
│   ├── draft_model.py            # Main draft model
│   └── transfer_draft_model.py   # Transfer-capable version
└── training/
    ├── train_single_set.py       # Train on one set
    ├── train_multi_set.py        # Train on multiple sets
    └── train_transfer.py         # Transfer learning training
```

## Conclusion

**v1 architecture (keyword-based) is NOT transfer-learning friendly** because:
1. Card embeddings are tied to specific card names
2. Fixed keyword vocabulary breaks on new mechanics
3. Parameterized abilities lose quantity information

**v2 architecture (text embeddings) WILL enable transfer** because:
1. Sentence transformers understand new text automatically
2. "Mill 3" vs "Mill 5" get different embeddings
3. 2024 research shows 42.87% accuracy on unseen cards

**Action Items** (see CARD_ENCODING.md for full details):
1. Integrate `text_embeddings.py` into `shared_card_encoder.py`
2. Pre-compute embeddings for all Standard cards
3. Train and compare v1 vs v2 models

The infrastructure is ready - we just need to update the encoder architecture.
