#!/usr/bin/env python3
"""Generate network architecture diagram for ForgeRL.

Creates a clean flowchart showing the full AlphaZero-style architecture:
- Card embedding MLP (shared across all encoders)
- 4 zone encoders (hand, battlefield, graveyard, exile) with self-attention
- Stack encoder with positional encoding
- Global encoder (life, mana, turn, phase)
- Cross-zone attention (3 layers)
- Policy head (3 layers → 203 actions)
- Value head (2 layers → win probability)

Outputs PNG and PDF to data/reports/network_architecture.*

Usage:
    python3 scripts/generate_network_diagram.py
    # Outputs: data/reports/network_architecture.png
    #          data/reports/network_architecture.pdf
"""

import sys
from pathlib import Path

try:
    import graphviz
except ImportError:
    print("ERROR: graphviz not installed")
    print("Install with: uv sync --extra dev")
    print("Also requires system graphviz: brew install graphviz (macOS)")
    sys.exit(1)

# Try to import architecture components to get exact parameter counts
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from forge.game_state_encoder import ForgeGameStateEncoder, GameStateConfig
    from forge.policy_value_heads import PolicyHead, ValueHead, PolicyValueConfig
    HAS_TORCH = True
except ImportError:
    # Torch not available, will use config values only
    HAS_TORCH = False

    # Define minimal config classes for diagram generation
    class GameStateConfig:
        vocab_size = 1387
        max_params = 37
        d_model = 512
        n_heads = 8
        n_layers = 3
        d_ff = 1024
        dropout = 0.1
        zone_embedding_dim = 512
        global_embedding_dim = 192
        output_dim = 768

    class PolicyValueConfig:
        state_dim = 768
        policy_hidden_dim = 384
        policy_n_layers = 3
        value_hidden_dim = 384
        value_n_layers = 2
        action_dim = 203


def count_params(module) -> int:
    """Count trainable parameters in a PyTorch module."""
    if not HAS_TORCH:
        return 0
    return sum(p.numel() for p in module.parameters())


def format_params(count: int) -> str:
    """Format parameter count in K/M."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.0f}K"
    else:
        return str(count)


def create_network_diagram(output_path: str = "data/reports/network_architecture"):
    """Generate the network architecture diagram."""

    # Get architecture config
    config = GameStateConfig()
    pv_config = PolicyValueConfig()

    # Create encoder to get param counts
    if HAS_TORCH:
        try:
            encoder = ForgeGameStateEncoder(config)
            policy = PolicyHead(pv_config)
            value = ValueHead(pv_config, num_players=2)

            # Count parameters per component
            card_emb_params = count_params(encoder.card_embedding)
            zone_enc_params = count_params(encoder.zone_encoders["hand"])
            stack_enc_params = count_params(encoder.stack_encoder)
            global_enc_params = count_params(encoder.global_encoder)
            cross_attn_params = count_params(encoder.cross_zone_attn)
            combine_params = count_params(encoder.combine)
            policy_params = count_params(policy)
            value_params = count_params(value)
            total_params = count_params(encoder) + policy_params + value_params
        except Exception as e:
            print(f"Warning: Could not instantiate network: {e}")
            print("Using config values only (no parameter counts)")
            card_emb_params = 0
            zone_enc_params = 0
            stack_enc_params = 0
            global_enc_params = 0
            cross_attn_params = 0
            combine_params = 0
            policy_params = 0
            value_params = 0
            total_params = 0
    else:
        print("Torch not available, using config values only (no parameter counts)")
        card_emb_params = 0
        zone_enc_params = 0
        stack_enc_params = 0
        global_enc_params = 0
        cross_attn_params = 0
        combine_params = 0
        policy_params = 0
        value_params = 0
        total_params = 0

    # Create directed graph
    dot = graphviz.Digraph(
        comment='ForgeRL Network Architecture',
        format='png',
        graph_attr={
            'rankdir': 'TB',
            'splines': 'ortho',
            'nodesep': '0.6',
            'ranksep': '0.8',
            'bgcolor': 'white',
            'fontname': 'Arial',
        },
        node_attr={
            'shape': 'box',
            'style': 'rounded,filled',
            'fontname': 'Arial',
            'fontsize': '10',
            'height': '0.5',
        },
        edge_attr={
            'fontname': 'Arial',
            'fontsize': '9',
            'arrowsize': '0.7',
        }
    )

    # Input layer
    dot.node('input',
             f'Card Features\n({config.vocab_size + config.max_params + 32}d)\n'
             f'mechanics + params + state',
             fillcolor='#E3F2FD', fontcolor='black')

    # Shared card embedding MLP (blue)
    params_str = f'\n{format_params(card_emb_params)} params' if card_emb_params else ''
    dot.node('card_emb',
             f'Shared Card MLP\n'
             f'{config.vocab_size + config.max_params + 32}→1024→{config.d_model}\n'
             f'GELU + LayerNorm{params_str}',
             fillcolor='#2196F3', fontcolor='white', fontsize='10')

    dot.edge('input', 'card_emb')

    # Zone encoders (4 parallel boxes in blue)
    zone_params_str = f'\n{format_params(zone_enc_params)} params each' if zone_enc_params else ''
    zones = ['hand', 'battlefield', 'graveyard', 'exile']
    zone_nodes = []

    with dot.subgraph(name='cluster_zones') as c:
        c.attr(label='Zone Encoders (parallel)', fontsize='11', style='dashed', color='gray')
        for zone in zones:
            node_id = f'zone_{zone}'
            zone_nodes.append(node_id)
            c.node(node_id,
                   f'{zone.capitalize()}\n'
                   f'2 self-attn layers\n'
                   f'{config.d_model}d → {config.zone_embedding_dim}d{zone_params_str}',
                   fillcolor='#1976D2', fontcolor='white', fontsize='9')
            dot.edge('card_emb', node_id, style='dashed')

    # Stack encoder (alongside zones, blue)
    stack_params_str = f'\n{format_params(stack_enc_params)} params' if stack_enc_params else ''
    dot.node('stack_enc',
             f'Stack Encoder\n'
             f'Positional encoding\n'
             f'Self-attention\n'
             f'{config.d_model}d → {config.zone_embedding_dim}d{stack_params_str}',
             fillcolor='#1976D2', fontcolor='white', fontsize='9')
    dot.edge('card_emb', 'stack_enc', style='dashed')

    # Global encoder (alongside zones, blue)
    global_params_str = f'\n{format_params(global_enc_params)} params' if global_enc_params else ''
    dot.node('global_enc',
             f'Global Encoder\n'
             f'Life, mana, turn, phase\n'
             f'Multi-layer MLP\n'
             f'→ {config.global_embedding_dim}d{global_params_str}',
             fillcolor='#1976D2', fontcolor='white', fontsize='9')

    # Cross-zone attention (green)
    attn_params_str = f'\n{format_params(cross_attn_params)} params' if cross_attn_params else ''
    dot.node('cross_attn',
             f'Cross-Zone Attention\n'
             f'3 layers × {config.n_heads} heads\n'
             f'{config.zone_embedding_dim}d{attn_params_str}',
             fillcolor='#4CAF50', fontcolor='white', fontsize='10')

    for zone_node in zone_nodes:
        dot.edge(zone_node, 'cross_attn')

    # Combine layer (green)
    combine_input_dim = (4 * config.zone_embedding_dim +  # zones
                        config.zone_embedding_dim +       # stack
                        config.global_embedding_dim)      # global
    combine_params_str = f'\n{format_params(combine_params)} params' if combine_params else ''
    dot.node('combine',
             f'Combine\n'
             f'{combine_input_dim}d → {config.d_ff}d → {config.output_dim}d\n'
             f'LayerNorm + GELU{combine_params_str}',
             fillcolor='#66BB6A', fontcolor='white', fontsize='10')

    dot.edge('cross_attn', 'combine')
    dot.edge('stack_enc', 'combine')
    dot.edge('global_enc', 'combine')

    # State embedding output
    dot.node('state_emb',
             f'State Embedding\n({config.output_dim}d)',
             fillcolor='#E8F5E9', fontcolor='black')
    dot.edge('combine', 'state_emb')

    # Policy head (orange)
    policy_params_str = f'\n{format_params(policy_params)} params' if policy_params else ''
    dot.node('policy_head',
             f'Policy Head\n'
             f'3 layers × {pv_config.policy_hidden_dim}d\n'
             f'GELU + LayerNorm + Dropout\n'
             f'→ {pv_config.action_dim} actions{policy_params_str}',
             fillcolor='#FF9800', fontcolor='white', fontsize='10')
    dot.edge('state_emb', 'policy_head')

    # Value head (orange)
    value_params_str = f'\n{format_params(value_params)} params' if value_params else ''
    dot.node('value_head',
             f'Value Head\n'
             f'2 layers × {pv_config.value_hidden_dim}d\n'
             f'GELU + LayerNorm + Dropout\n'
             f'→ win probability{value_params_str}',
             fillcolor='#F57C00', fontcolor='white', fontsize='10')
    dot.edge('state_emb', 'value_head')

    # Output nodes
    dot.node('policy_out',
             f'Action Probabilities\n({pv_config.action_dim}d)',
             fillcolor='#FFE0B2', fontcolor='black')
    dot.node('value_out',
             'Win Probability\n(1d)',
             fillcolor='#FFE0B2', fontcolor='black')

    dot.edge('policy_head', 'policy_out')
    dot.edge('value_head', 'value_out')

    # Add title and param count
    total_str = f'{format_params(total_params)} total params' if total_params else 'Network Architecture'
    dot.attr(label=f'ForgeRL AlphaZero Architecture\n{total_str}',
             labelloc='t', fontsize='14', fontname='Arial Bold')

    # Render
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render PNG
    dot.render(output_path, format='png', cleanup=True)
    print(f"Generated: {output_path}.png")

    # Render PDF
    dot.render(output_path, format='pdf', cleanup=True)
    print(f"Generated: {output_path}.pdf")

    if total_params:
        print(f"\nNetwork parameters: {format_params(total_params)}")
        print(f"  Card embedding:   {format_params(card_emb_params)}")
        print(f"  Zone encoder:     {format_params(zone_enc_params)} × 4")
        print(f"  Stack encoder:    {format_params(stack_enc_params)}")
        print(f"  Global encoder:   {format_params(global_enc_params)}")
        print(f"  Cross-zone attn:  {format_params(cross_attn_params)}")
        print(f"  Combine:          {format_params(combine_params)}")
        print(f"  Policy head:      {format_params(policy_params)}")
        print(f"  Value head:       {format_params(value_params)}")


if __name__ == "__main__":
    create_network_diagram()
