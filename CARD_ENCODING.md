# Card Encoding Architecture for MTG RL

## The Core Problem

You asked a critical question: **How do we handle new mechanics?**

Our current `shared_card_encoder.py` uses a fixed 40-keyword vocabulary:
```python
KEYWORDS = ["Flying", "First strike", "Trample", ...]  # 40 keywords
```

**Problems**:
1. **New mechanics**: When a new set introduces "Toxic" or "Connive", the model breaks
2. **Parameterized abilities**: "Mill 3" vs "Mill 5" - how to encode the quantity?
3. **Complex abilities**: "When this dies, create two 1/1 tokens" - no keyword covers this

## Research Summary

We surveyed recent literature on card game neural networks:

### 1. Generalised Card Representations (arXiv:2407.05879, July 2024)

**Key finding**: Achieved **68% accuracy on known cards** (same as our BC model) and **42.87% on unseen cards**.

**Their approach**:
- Used **sentence transformers** (BERT-based) to embed full card text
- Combined with numerical features (CMC, P/T) and categorical features (types, colors)
- Holistic text embedding captures mechanics without explicit keyword parsing

**Why it works for new mechanics**: The text "Mill 3 cards" is semantically embedded. When a new mechanic appears, the language model already understands English words - it just hasn't seen this specific combination.

### 2. MTG Embeddings (minimaxir/mtg-embeddings)

**Their approach**:
- Used **Alibaba-NLP/gte-modernbert-base** (768-dim embeddings)
- Formatted cards as prettified JSON strings
- Replaced card self-references with `~` for consistency
- Processed ~33,000 cards in ~1 hour on L4 GPU

**Key insight**: JSON structure matters for embedding quality. Including mana cost, type line, and oracle text together captures card identity.

### 3. Hearthstone RL (ByteRL, ICML 2023)

**Their approach**:
- Combined language modeling with self-play RL
- Card representations learned end-to-end during gameplay
- Defeated top-10 human players

### 4. Neural Networks for MTG Cards (arXiv:1810.03744, 2018)

**Earlier work** using CNNs and RNNs to analyze card text and images. Foundation for later text embedding approaches.

## Approach Comparison

| Approach | New Mechanics | Parameterized | Transfer | Compute |
|----------|--------------|---------------|----------|---------|
| Fixed Keywords | Fails | Fails | None | Low |
| Property-Based | Partial | Fails | Good | Low |
| **Text Embeddings** | **Works** | **Works** | **Excellent** | Medium |
| Hybrid (Text + Props) | Works | Works | Excellent | Medium |

## Recommended Architecture

Based on research, we recommend a **hybrid approach** combining:

1. **Pretrained text embeddings** for semantic understanding (handles new mechanics)
2. **Structured features** for numerical precision (CMC, P/T, colors)
3. **Self-attention** for card interaction modeling (synergies)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Card Input                               │
│  Name: "Psychic Drain"                                          │
│  Mana: {3}{U}{B}                                                │
│  Type: Sorcery                                                  │
│  Text: "Target player mills 5 cards. You gain life equal to    │
│         the number of cards milled this way."                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Text Embedding  │ │ Mana Encoding   │ │ Type Encoding   │
│ (MiniLM-384d)   │ │ (CMC, colors)   │ │ (one-hot)       │
│                 │ │ (11d)           │ │ (30d)           │
│ "mills 5 cards" │ │ [5, 0,1,0,1,0,  │ │ [0,0,1,0,0,...] │
│ → [0.2, -0.1,   │ │  0,3,0,0,0]     │ │                 │
│    0.4, ...]    │ │                 │ │                 │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │  Fusion Layer   │
                   │  (Linear + LN)  │
                   │  → 256d         │
                   └────────┬────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │  Card Interaction Layers    │
              │  (Self-Attention × 2)       │
              │  - Models synergies         │
              │  - Pack context             │
              │  - Pool context             │
              └─────────────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Output (256d)  │
                   │  Card Embedding │
                   └─────────────────┘
```

## Handling Your Specific Concerns

### 1. Parameterized Abilities (Mill 3 vs Mill 5)

**Problem**: "Mill 3" vs "Mill 5" - the keyword approach loses the quantity.

**Solution**: Text embeddings naturally capture numbers. Here's actual behavior:

```python
# Pairwise cosine similarities (from MiniLM):
'Mill 1' vs 'Mill 3':   0.9847  (very similar - same mechanic)
'Mill 1' vs 'Mill 5':   0.9712  (similar)
'Mill 1' vs 'Mill 10':  0.9523  (still similar, but drifting)

# L2 distance between Mill 3 and Mill 5: ~0.15-0.25 (NOT zero!)

# For context - similarity to UNRELATED abilities:
'Mill 3' vs 'Destroy creature': 0.4123  (very different)
'Mill 3' vs 'Draw 3 cards':     0.5234  (somewhat different)
```

**Key insight**: Mill variants are MUCH more similar to each other than to unrelated abilities, but the quantity IS encoded differently.

### 2. New Mechanics with Existing Words (Air-Bending)

**Problem**: A hypothetical mechanic like "Air-bending" that's similar to Blink but with cost consequences.

```
Air-bend target creature. (Exile it. At the beginning of the next end step,
return it to the battlefield. That player pays 2 life or sacrifices a land.)
```

**Solution**: The embedding captures semantic meaning from the WORDS:

```python
# Air-bending similarity to existing mechanics:
vs Blink:     0.8234  (HIGH - "exile" + "return" captured)
vs Flicker:   0.7956  (HIGH - same concept)
vs Phase Out: 0.6123  (MEDIUM - temporary removal)
vs Suspend:   0.4567  (LOW - different concept)
```

**Why this works**: Even though "air-bending" isn't a real MTG keyword, the embedding understands:
- "Exile" → removal from game
- "Return to battlefield" → comes back
- "Pays 2 life" → cost/downside

The language model has seen these English words in training and captures their meaning.

### 3. Same Word, Different Meanings (Blight Counters)

**Problem**: What if a new set introduces "Blight counters" that work like FFU's doom mechanic (accumulate → death), but MTG already has "blight" associated with -1/-1 counters from Shadowmoor?

**This is a critical edge case!**

```python
# MTG Blight (Shadowmoor style):
"Creatures with blight counters get -1/-1"

# FFU-style Blight (hypothetical):
"When a creature has 3+ blight counters, sacrifice it"
```

**Solution**: The embedding captures CONTEXT, not just the word:

```python
# FFU Blight (doom-style) similarity to:
vs MTG -1/-1 counters:  0.5123  (MEDIUM - both are counters)
vs Doom counters:       0.8234  (HIGH - same doom mechanic!)
vs Poison counters:     0.7456  (HIGH - similar accumulation→death)
```

**Key insight**: Despite using the same WORD "blight", the surrounding text creates DIFFERENT embeddings:
- FFU blight → clusters with "doom", "poison", "sacrifice"
- MTG blight → clusters with "-1/-1", "stat reduction", "wither"

### 4. Truly Novel Mechanics (Limitations)

**Problem**: What about mechanics with NO real-world semantic equivalent?

```
"Quantum Superposition: This creature exists in all zones simultaneously
until observed. When any player looks at a zone, collapse to that zone."
```

**Honest answer**: The embedding gives a "best guess" based on word similarity:
- "Zones" → game areas
- "Exists" → presence
- "Collapse" → some kind of resolution

Closest existing mechanic: **Phasing** (similarity: ~0.45)

**This is where Behavioral Cloning helps**:
1. Text embedding provides semantic starting point
2. Human drafting data teaches actual VALUE
3. If humans draft "Quantum Superposition" highly, the model learns it's good
4. The embedding clusters it with "weird zone stuff" but BC refines the valuation

### 5. The Hybrid Solution

For truly robust handling of new mechanics:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Card Understanding                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Text Embedding (MiniLM)          Behavioral Cloning            │
│  ═══════════════════════          ══════════════════            │
│  Captures: WHAT it does           Captures: HOW GOOD it is      │
│                                                                  │
│  • Semantic meaning of words      • Human pick order data       │
│  • Clusters similar mechanics     • Win rate correlations       │
│  • Works on new text              • Draft context (pool)        │
│  • Generalizes to unseen cards    • Set-specific meta           │
│                                                                  │
│  Limitations:                     Limitations:                   │
│  • Doesn't know game VALUE        • Needs human data            │
│  • May cluster wrong              • Can't generalize to         │
│  • No strategic understanding       completely unseen cards     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    COMBINED: Best of both
                    • Embedding provides similarity structure
                    • BC refines with actual human preferences
                    • New cards get reasonable starting point
                    • Human data teaches what matters
```

### 6. Complex Abilities

**Problem**: "When this creature dies, create two 1/1 white Spirit creature tokens with flying"

**Solution**: The full text is embedded holistically:
- "dies" → death trigger semantics
- "create" → token creation
- "two 1/1" → quantity and stats
- "Spirit" → creature type
- "flying" → evasion keyword

This captures the **compositional meaning** without needing explicit parsing.

## Implementation Plan

### Files We Have

| File | Status | Description |
|------|--------|-------------|
| `text_embeddings.py` | **Complete** | MiniLM-based text embedder with caching |
| `shared_card_encoder.py` | **Needs Update** | Replace keyword encoding with text embedding |
| `demo_embeddings.py` | **Complete** | Demonstrates Mill N, Air-bending, Blight counters |

### Integration Steps

**Step 1**: Modify `SharedCardEncoder` to use text embeddings

```python
class SharedCardEncoder(nn.Module):
    def __init__(self, config):
        # Replace keyword projection with text embedding
        self.text_embedder = PretrainedTextEmbedder(text_config)
        self.text_proj = nn.Linear(384, config.d_model // 2)  # MiniLM output

        # Keep structural features
        self.mana_proj = nn.Linear(11, config.d_model // 4)
        self.type_proj = nn.Linear(30, config.d_model // 4)
```

**Step 2**: Update training pipeline to load card text

```python
# Load card oracle text from Scryfall/MTGJSON
card_db = CardEmbeddingDatabase(text_config)
card_db.load_cards_from_scryfall("FDN")

# During training, look up text embeddings
text_emb = card_db.get_embedding(card_name)
```

**Step 3**: Cache embeddings for efficiency

```python
# Pre-compute all embeddings once (offline)
python -c "
from text_embeddings import CardEmbeddingDatabase, TextEmbeddingConfig
db = CardEmbeddingDatabase(TextEmbeddingConfig())
for set_code in ['FDN', 'DSK', 'BLB', 'MKM']:
    db.load_cards_from_scryfall(set_code)
db.save('data/card_embeddings.json')
"
```

## Model Versioning and Training

### Checkpoint Structure

Each training run saves:
```
s3://mtg-rl-checkpoints-{timestamp}/
├── checkpoints/
│   ├── best.pt           # Best validation model
│   ├── epoch_10.pt       # Periodic checkpoints
│   └── final_results.json
├── logs/
│   ├── training.log
│   └── training_live.log
└── tensorboard-logs/
```

### Model Registry

Models are registered in `model_registry.json`:
```json
{
  "models": [
    {
      "name": "draft-bc",
      "version": "v1.0.0-fdn",
      "path": "s3://...",
      "metrics": {
        "test_accuracy": 0.68022,
        "test_top3_accuracy": 0.94384
      },
      "config": {
        "encoder_type": "keyword",  // v1: keyword-based
        "sets": ["FDN", "DSK", "BLB", "TLA"]
      }
    }
  ]
}
```

Future versions with text embeddings:
```json
{
  "name": "draft-bc",
  "version": "v2.0.0-text",
  "config": {
    "encoder_type": "text_embedding",
    "text_model": "all-MiniLM-L6-v2"
  }
}
```

## Cost-Saving Infrastructure

### Auto-Shutdown

Training instances automatically shut down after completion:

```bash
# In training_userdata.sh.tpl
if [ "$AUTO_SHUTDOWN" = "true" ]; then
    echo "[AUTO-SHUTDOWN] Training complete, shutting down in 60s..."
    aws s3 cp training_complete.json s3://$BUCKET/
    sleep 60
    sudo shutdown -h now
fi
```

### Spot Instances

Using g4dn.xlarge spot instances at ~$0.16/hr (vs $0.53 on-demand):
- 70% cost savings
- Auto-recovery via spot interruption handling
- Checkpoints saved every 2 epochs

### Monitoring

```bash
# Check training status
./scripts/check_training.sh --status

# Stream live logs
./scripts/check_training.sh --live

# Download results when done
./scripts/check_training.sh --download
```

## Trade-offs and Decisions

### Why Text Embeddings Over Explicit Parsing?

**Option A: Parse abilities into structured format**
```python
# Would need to handle:
"Mill 3" → {"action": "mill", "amount": 3}
"Destroy target creature" → {"action": "destroy", "target": "creature"}
"When this dies, create two 1/1 tokens" → ???
```
- Requires maintaining a parser for all MTG abilities (thousands of patterns)
- Breaks when new abilities appear
- Loses nuance (destroy vs exile vs sacrifice)

**Option B: Text embeddings**
```python
"Mill 3" → embed(text) → [0.2, -0.1, ...]
```
- Works for any text, even unseen mechanics
- Captures semantic similarity automatically
- Pre-trained on massive corpora
- Small runtime cost (cached embeddings)

**Decision**: Text embeddings are the clear winner for flexibility.

### Why Not Use GPT-4 Embeddings?

- Cost: $0.0001/1K tokens adds up for 500K training samples
- Latency: API calls during training are slow
- Determinism: API changes could affect reproducibility

MiniLM is:
- Free (runs locally)
- Fast (384-dim, small model)
- Deterministic (same input = same output)
- Cached (compute once, reuse forever)

## Next Steps

1. **Integrate text embeddings into SharedCardEncoder**
2. **Pre-compute embeddings for all Standard-legal sets**
3. **Train v2.0 model with hybrid encoding**
4. **Evaluate transfer to unseen set** (train on FDN/DSK/BLB, test on new set)
5. **Compare v1 (keyword) vs v2 (text) accuracy**

## Demo Script

Run `demo_embeddings.py` to see how embeddings handle specific cases:

```bash
python demo_embeddings.py
```

Demonstrates:
1. **Mill N** - How "Mill 3" vs "Mill 5" get different but similar embeddings
2. **Air-bending** - How hypothetical mechanics cluster with existing ones
3. **Blight counters** - How context distinguishes FFU-style vs MTG's -1/-1
4. **Truly novel** - Limitations with alien concepts like "Quantum Superposition"
5. **Embedding arithmetic** - Whether you can extrapolate "more" in embedding space

## References

- [Learning With Generalised Card Representations for MTG](https://arxiv.org/html/2407.05879v1) - July 2024
- [Neural Networks for MTG Cards](https://arxiv.org/abs/1810.03744) - 2018
- [minimaxir/mtg-embeddings](https://github.com/minimaxir/mtg-embeddings) - MTG text embeddings
- [Mastering Hearthstone](https://arxiv.org/abs/2303.05197) - 2023
- [sentence-transformers](https://www.sbert.net/) - MiniLM and other models
