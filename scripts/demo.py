#!/usr/bin/env python3
"""
MTG RL Demo

Quick demonstration of the complete MTG RL pipeline:
1. Card embeddings
2. Policy network
3. Training (mock)
4. Evaluation (mock)
"""


import numpy as np
import torch

from src.models.card_embeddings import CardEmbedding
from src.utils.evaluate import GameResult, EvalMetrics
from src.models.policy_network import MTGPolicyNetwork, TransformerConfig
from src.agents.ppo_agent import PPOAgent, PPOConfig
print("="*60)
print("MTG Reinforcement Learning Demo")
print("="*60)

# =============================================================================
# 1. Card Embeddings Demo
# =============================================================================
print("\n1. CARD EMBEDDINGS")
print("-"*60)

embedder = CardEmbedding(use_text_embeddings=False)

# Create some sample cards
cards = [
    {
        'name': 'Lightning Bolt',
        'mana_cost': 'R',
        'types': 'Instant',
        'oracle_text': 'Lightning Bolt deals 3 damage to any target.',
        'keywords': [],
        'power': 0, 'toughness': 0
    },
    {
        'name': 'Goblin Guide',
        'mana_cost': 'R',
        'types': 'Creature — Goblin Scout',
        'oracle_text': 'Haste. Whenever Goblin Guide attacks, defending player reveals the top card of their library.',
        'keywords': ['Haste'],
        'power': 2, 'toughness': 2
    },
    {
        'name': 'Counterspell',
        'mana_cost': 'UU',
        'types': 'Instant',
        'oracle_text': 'Counter target spell.',
        'keywords': [],
        'power': 0, 'toughness': 0
    },
    {
        'name': 'Serra Angel',
        'mana_cost': '3WW',
        'types': 'Creature — Angel',
        'oracle_text': 'Flying, vigilance',
        'keywords': ['Flying', 'Vigilance'],
        'power': 4, 'toughness': 4
    },
]

print(f"Embedding dimension: {embedder.total_dim}")
print("\nCard embeddings:")

embeddings = {}
for card in cards:
    emb = embedder.embed_from_game_state(card)
    embeddings[card['name']] = emb
    print(f"  {card['name']}: shape={emb.shape}, norm={np.linalg.norm(emb):.3f}")

# Show similarity
print("\nCard similarities (cosine):")
for i, c1 in enumerate(cards):
    for c2 in cards[i+1:]:
        e1, e2 = embeddings[c1['name']], embeddings[c2['name']]
        sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
        print(f"  {c1['name']} <-> {c2['name']}: {sim:.3f}")

# =============================================================================
# 2. Policy Network Demo
# =============================================================================
print("\n2. POLICY NETWORK")
print("-"*60)

config = TransformerConfig()
network = MTGPolicyNetwork(config)

total_params = sum(p.numel() for p in network.parameters())
print(f"Network parameters: {total_params:,}")
print("Architecture:")
print(f"  - Card embedding dim: {config.card_embedding_dim}")
print(f"  - Model dimension: {config.d_model}")
print(f"  - Attention heads: {config.n_heads}")
print(f"  - Transformer layers: {config.n_layers}")
print(f"  - Max actions: {config.max_actions}")

# Test forward pass
batch_size = 1
card_emb = torch.randn(batch_size, config.max_sequence_length, config.card_embedding_dim)
zone_ids = torch.randint(0, 6, (batch_size, config.max_sequence_length))
card_mask = torch.zeros(batch_size, config.max_sequence_length)
card_mask[:, :10] = 1.0
global_feat = torch.randn(batch_size, config.global_feature_dim)
action_mask = torch.ones(batch_size, config.max_actions)
action_mask[:, 5:] = 0  # Only 5 valid actions

network.eval()
with torch.no_grad():
    logits, probs, value = network(card_emb, zone_ids, card_mask, global_feat, action_mask)

print("\nForward pass:")
print(f"  Action probabilities: {probs[0, :5].numpy()}")
print(f"  State value: {value.item():.4f}")

# =============================================================================
# 3. PPO Agent Demo
# =============================================================================
print("\n3. PPO AGENT")
print("-"*60)

ppo_config = PPOConfig(n_steps=16)
agent = PPOAgent(ppo_config, config, torch.device('cpu'))

print("PPO hyperparameters:")
print(f"  - Learning rate: {ppo_config.learning_rate}")
print(f"  - Gamma: {ppo_config.gamma}")
print(f"  - GAE lambda: {ppo_config.gae_lambda}")
print(f"  - Clip epsilon: {ppo_config.clip_epsilon}")

# Mock game state
mock_state = {
    'turn': 5,
    'our_player': {
        'life': 18,
        'hand': [cards[0], cards[1]],
        'battlefield': [cards[3]],
        'graveyard': [],
    },
    'opponent': {
        'life': 15,
        'hand': [],
        'battlefield': [],
        'graveyard': [],
    },
    'stack': [],
}

action_mask = np.zeros(50, dtype=np.float32)
action_mask[:3] = 1.0  # 3 valid actions

action, info = agent.get_action(mock_state, action_mask, deterministic=True)
print("\nAction selection:")
print(f"  Selected action: {action}")
print(f"  Value estimate: {info['value'].item():.4f}")
print(f"  Action probs: {info['action_probs'][0, :5].numpy()}")

# =============================================================================
# 4. Training Loop Demo
# =============================================================================
print("\n4. TRAINING LOOP (Mock)")
print("-"*60)

# Collect fake experiences
print("Collecting experiences...")
for step in range(16):
    mock_state['turn'] = 5 + step
    action, info = agent.get_action(mock_state, action_mask)
    reward = np.random.randn() * 0.1
    done = step == 15
    agent.store_transition(info, action, reward, done)

# Train
print("Training...")
last_value = torch.tensor([0.0])
metrics = agent.train_step(last_value)

print("\nTraining metrics:")
print(f"  Policy loss: {metrics['policy_loss']:.6f}")
print(f"  Value loss: {metrics['value_loss']:.6f}")
print(f"  Entropy: {metrics['entropy']:.6f}")
print(f"  Approx KL: {metrics['approx_kl']:.6f}")

# =============================================================================
# 5. Evaluation Demo
# =============================================================================
print("\n5. EVALUATION (Mock)")
print("-"*60)

# Generate mock game results
results = []
for i in range(50):
    results.append(GameResult(
        game_id=i+1,
        won=np.random.random() > 0.35,  # ~65% win rate
        game_length=np.random.randint(5, 20),
        decisions_made=np.random.randint(20, 100),
        our_final_life=np.random.randint(0, 20),
        opponent_final_life=np.random.randint(0, 20),
        duration_seconds=np.random.uniform(10, 60),
        deck_matchup="red_aggro.dck vs white_weenie.dck",
        opponent_type="ai"
    ))

metrics = EvalMetrics()
metrics.update(results)

print("Evaluation results (50 mock games):")
print(f"  Win rate: {metrics.win_rate*100:.1f}% (+/- {metrics.win_rate_std*100:.1f}%)")
print(f"  Avg game length: {metrics.avg_game_length:.1f} turns")
print(f"  Avg decisions: {metrics.avg_decisions:.1f}")
print(f"  Avg life differential: {metrics.avg_final_life_diff:+.1f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*60)
print("DEMO COMPLETE")
print("="*60)
print("""
The MTG RL system is ready for training!

Files created:
  - card_embeddings.py  : Card embedding system (89 dims)
  - policy_network.py   : Transformer policy network (2.6M params)
  - ppo_agent.py        : PPO training agent
  - rl_environment.py   : Forge environment wrapper
  - train.py            : Training script (parallel/single)
  - evaluate.py         : Evaluation framework

To train with Forge:
  1. Build Docker image: docker build -f infrastructure/docker/Dockerfile.sim -t forge-sim .
  2. Run training: python3 train.py --mode single --timesteps 10000
  3. Evaluate: python3 evaluate.py --mode agent --agent checkpoints/final_model.pt

To test without Docker:
  python3 train.py --mode test
  python3 evaluate.py --mode test
""")
