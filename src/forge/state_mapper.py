"""
State Mapper

Maps Forge daemon game state to our neural network encoder format.
This bridges the gap between the JSON protocol and the tensor encoding.

The ForgeGameStateEncoder expects structured input about zones, mana, life, etc.
This module converts the JSON decision format to that structure.
"""

import torch
from typing import Optional
from dataclasses import dataclass, field

from src.forge.forge_client import (
    ActionOption,
    Decision,
    GameState,
    PlayerState,
    CardInfo,
    DecisionType,
)
from src.forge.game_state_encoder import (
    ForgeGameStateEncoder,
    GameStateConfig,
    Phase,
)


# Phase mapping from Forge daemon to our encoder
PHASE_MAP = {
    "UNTAP": Phase.UNTAP,
    "UPKEEP": Phase.UPKEEP,
    "DRAW": Phase.DRAW,
    "MAIN1": Phase.MAIN1,
    "COMBAT_BEGIN": Phase.COMBAT_BEGIN,
    "COMBAT_DECLARE_ATTACKERS": Phase.COMBAT_ATTACKERS,
    "COMBAT_DECLARE_BLOCKERS": Phase.COMBAT_BLOCKERS,
    "COMBAT_FIRST_STRIKE_DAMAGE": Phase.COMBAT_FIRST_STRIKE,
    "COMBAT_DAMAGE": Phase.COMBAT_DAMAGE,
    "COMBAT_END": Phase.COMBAT_END,
    "MAIN2": Phase.MAIN2,
    "END_OF_TURN": Phase.END,
    "CLEANUP": Phase.CLEANUP,
}


@dataclass
class EncodedState:
    """Encoded game state ready for neural network."""
    state_tensor: torch.Tensor  # [state_dim] or [batch, state_dim]
    action_mask: torch.Tensor  # [num_actions] or [batch, num_actions]
    our_player_idx: int
    is_our_turn: bool
    legal_action_indices: list[int] = field(default_factory=list)
    action_descriptions: list[str] = field(default_factory=list)


class StateMapper:
    """
    Maps Forge game state to neural network input format.

    Usage:
        mapper = StateMapper()

        # Get encoder output
        encoded = mapper.encode_decision(decision)

        # Use with network
        policy, value = network(encoded.state_tensor, encoded.action_mask)
    """

    def __init__(self, config: Optional[GameStateConfig] = None):
        self.config = config or GameStateConfig()
        self.encoder = ForgeGameStateEncoder(self.config)

        # Card name to mechanics embedding cache
        self._card_embedding_cache: dict[str, torch.Tensor] = {}

    def encode_decision(
        self,
        decision: Decision,
        our_player_name: str,
    ) -> EncodedState:
        """
        Encode a decision request for neural network processing.

        Args:
            decision: Decision from Forge daemon
            our_player_name: Name of our player

        Returns:
            EncodedState with tensors ready for network
        """
        game_state = decision.game_state

        # Find our player
        our_player = None
        opponent = None
        our_idx = 0
        for i, p in enumerate(game_state.players):
            if p.name == our_player_name or our_player_name in p.name:
                our_player = p
                our_idx = i
            else:
                opponent = p

        if our_player is None:
            raise ValueError(f"Could not find player {our_player_name} in game state")
        if opponent is None and len(game_state.players) > 1:
            opponent = game_state.players[1 - our_idx]

        # Build encoder inputs
        # Hand cards
        hand_cards = self._encode_card_list(our_player.hand)

        # Battlefield split by player
        our_battlefield = self._encode_card_list(our_player.battlefield)
        opp_battlefield = self._encode_card_list(opponent.battlefield) if opponent else []

        # Graveyards
        our_graveyard = self._encode_card_list(our_player.graveyard)
        opp_graveyard = self._encode_card_list(opponent.graveyard) if opponent else []

        # Exile
        exile_cards = self._encode_card_list(our_player.exile)
        if opponent:
            exile_cards.extend(self._encode_card_list(opponent.exile))

        # Convert to tensors
        hand_tensor = self._cards_to_tensor(hand_cards)
        our_bf_tensor = self._cards_to_tensor(our_battlefield)
        opp_bf_tensor = self._cards_to_tensor(opp_battlefield)
        our_gy_tensor = self._cards_to_tensor(our_graveyard)
        opp_gy_tensor = self._cards_to_tensor(opp_graveyard)
        exile_tensor = self._cards_to_tensor(exile_cards)

        # Scalar features
        phase = PHASE_MAP.get(decision.phase, Phase.MAIN)
        mana_available = self._encode_mana(our_player.mana_pool)

        # Encode using our encoder
        state_encoding = self.encoder.encode_game_state(
            hand=hand_tensor,
            our_battlefield=our_bf_tensor,
            opp_battlefield=opp_bf_tensor,
            our_graveyard=our_gy_tensor,
            opp_graveyard=opp_gy_tensor,
            exile=exile_tensor,
            our_life=torch.tensor([our_player.life], dtype=torch.float32),
            opp_life=torch.tensor([opponent.life if opponent else 20], dtype=torch.float32),
            mana_available=mana_available,
            phase=phase,
            is_our_turn=game_state.active_player == our_player_name,
            library_size=our_player.library_size,
            opp_library_size=opponent.library_size if opponent else 40,
            opp_hand_size=opponent.hand_size if opponent else 0,
        )

        # Create action mask based on decision type
        action_mask, legal_indices, descriptions = self._create_action_mask(decision)

        return EncodedState(
            state_tensor=state_encoding,
            action_mask=action_mask,
            our_player_idx=our_idx,
            is_our_turn=game_state.active_player == our_player_name,
            legal_action_indices=legal_indices,
            action_descriptions=descriptions,
        )

    def _encode_card_list(self, cards: list[CardInfo]) -> list[dict]:
        """Convert CardInfo list to encoder format."""
        result = []
        for card in cards:
            result.append({
                "id": card.id,
                "name": card.name,
                "cmc": card.cmc,
                "types": card.types.lower(),
                "oracle_text": card.oracle_text,
                "power": card.power or 0,
                "toughness": card.toughness or 0,
                "tapped": card.tapped,
                "summoning_sick": card.summoning_sick,
                "is_creature": card.is_creature or "creature" in card.types.lower(),
                "is_land": card.is_land or "land" in card.types.lower(),
            })
        return result

    def _cards_to_tensor(self, cards: list[dict]) -> torch.Tensor:
        """
        Convert card list to tensor.

        For now, use simple embedding. Later can integrate with text embeddings.
        Each card is represented as [cmc, power, toughness, is_creature, is_land, is_tapped]
        """
        card_dim = 16  # Dimension per card
        max_cards = self.config.max_hand_size  # Reuse config

        if not cards:
            return torch.zeros(max_cards, card_dim)

        features = []
        for card in cards[:max_cards]:
            feat = [
                card.get("cmc", 0) / 10.0,  # Normalize CMC
                card.get("power", 0) / 10.0,
                card.get("toughness", 0) / 10.0,
                1.0 if card.get("is_creature") else 0.0,
                1.0 if card.get("is_land") else 0.0,
                1.0 if card.get("tapped") else 0.0,
                1.0 if card.get("summoning_sick") else 0.0,
            ]
            # Pad to card_dim
            feat = feat + [0.0] * (card_dim - len(feat))
            features.append(feat)

        # Pad to max_cards
        while len(features) < max_cards:
            features.append([0.0] * card_dim)

        return torch.tensor(features, dtype=torch.float32)

    def _encode_mana(self, mana_pool) -> torch.Tensor:
        """Convert mana pool to tensor [W, U, B, R, G, C]."""
        return torch.tensor([
            mana_pool.white,
            mana_pool.blue,
            mana_pool.black,
            mana_pool.red,
            mana_pool.green,
            mana_pool.colorless,
        ], dtype=torch.float32)

    def _create_action_mask(
        self,
        decision: Decision,
    ) -> tuple[torch.Tensor, list[int], list[str]]:
        """
        Create action mask based on decision type and available actions.

        Returns:
            (action_mask, legal_indices, descriptions)
        """
        # Default action space size
        num_actions = 203  # From ActionConfig default (3+15+50*3+20+5+10)

        mask = torch.zeros(num_actions)
        legal_indices = []
        descriptions = []

        if decision.decision_type == DecisionType.CHOOSE_ACTION:
            # Map Forge actions to our action space
            for action in decision.actions:
                if action.index == -1:
                    # Pass action - typically index 0 in our space
                    idx = 0
                else:
                    # Map to action index (simplified)
                    # In full implementation, would map based on action type
                    idx = min(action.index + 1, num_actions - 1)

                mask[idx] = 1.0
                legal_indices.append(idx)
                descriptions.append(action.description)

        elif decision.decision_type == DecisionType.DECLARE_ATTACKERS:
            # Each attacker is a binary choice
            attackers = decision.raw_data.get("attackers", [])
            for i, attacker in enumerate(attackers[:num_actions]):
                mask[i] = 1.0
                legal_indices.append(i)
                descriptions.append(f"Attack with {attacker.get('name', 'creature')}")

            # Also allow attacking with none
            if attackers:
                mask[num_actions - 1] = 1.0  # "Attack with none" option
                legal_indices.append(num_actions - 1)
                descriptions.append("Attack with no creatures")

        elif decision.decision_type == DecisionType.DECLARE_BLOCKERS:
            # Blocking decisions
            blockers = decision.raw_data.get("blockers", [])
            for i, blocker in enumerate(blockers[:num_actions]):
                mask[i] = 1.0
                legal_indices.append(i)
                descriptions.append(f"Block with {blocker.get('name', 'creature')}")

            # Allow no blocks
            mask[num_actions - 1] = 1.0
            legal_indices.append(num_actions - 1)
            descriptions.append("Don't block")

        elif decision.decision_type in [
            DecisionType.PLAY_TRIGGER,
            DecisionType.CONFIRM_ACTION,
            DecisionType.PLAY_FROM_EFFECT,
        ]:
            # Binary yes/no
            mask[0] = 1.0  # Yes
            mask[1] = 1.0  # No
            legal_indices = [0, 1]
            descriptions = ["Yes", "No"]

        elif decision.decision_type == DecisionType.CHOOSE_CARDS:
            # Card selection
            cards = decision.raw_data.get("cards", [])
            for i, card in enumerate(cards[:num_actions]):
                mask[i] = 1.0
                legal_indices.append(i)
                descriptions.append(f"Choose {card.get('name', 'card')}")

        else:
            # Default: allow first action
            mask[0] = 1.0
            legal_indices.append(0)
            descriptions.append("Default action")

        return mask, legal_indices, descriptions


def action_index_to_response(
    action_idx: int,
    decision: Decision,
    encoded: EncodedState,
) -> str:
    """
    Convert neural network action index to Forge response string.

    Args:
        action_idx: Index from network output
        decision: Original decision
        encoded: Encoded state with action mapping

    Returns:
        Response string to send to Forge
    """
    if decision.decision_type == DecisionType.CHOOSE_ACTION:
        if action_idx == 0:
            return "-1"  # Pass
        # Map back to Forge action index
        forge_idx = action_idx - 1
        return str(forge_idx)

    elif decision.decision_type == DecisionType.DECLARE_ATTACKERS:
        attackers = decision.raw_data.get("attackers", [])
        if action_idx >= len(attackers):
            return ""  # No attackers
        # Return single attacker index (simplified)
        return str(action_idx)

    elif decision.decision_type == DecisionType.DECLARE_BLOCKERS:
        if action_idx >= len(decision.raw_data.get("blockers", [])):
            return ""  # No blockers
        return str(action_idx)

    elif decision.decision_type in [
        DecisionType.PLAY_TRIGGER,
        DecisionType.CONFIRM_ACTION,
        DecisionType.PLAY_FROM_EFFECT,
    ]:
        return "y" if action_idx == 0 else "n"

    elif decision.decision_type == DecisionType.CHOOSE_CARDS:
        return str(action_idx)

    else:
        return str(action_idx)


class ForgeNetworkAgent:
    """
    Agent that uses neural network to make decisions in Forge games.

    Usage:
        agent = ForgeNetworkAgent(network, mapper)
        client.start_game(deck1, deck2)

        while True:
            decision = client.receive_decision()
            if decision is None:
                break
            response = agent.decide(decision, our_player_name)
            client.send_response(response)
    """

    def __init__(
        self,
        network,  # PolicyValueNetwork or similar
        mapper: Optional[StateMapper] = None,
        temperature: float = 1.0,
    ):
        self.network = network
        self.mapper = mapper or StateMapper()
        self.temperature = temperature
        self.device = next(network.parameters()).device

    @torch.no_grad()
    def decide(self, decision: Decision, our_player_name: str) -> str:
        """
        Make a decision using the neural network.

        Args:
            decision: Decision from Forge
            our_player_name: Our player name

        Returns:
            Response string for Forge
        """
        # Encode state
        encoded = self.mapper.encode_decision(decision, our_player_name)

        # Move to device
        state = encoded.state_tensor.unsqueeze(0).to(self.device)
        mask = encoded.action_mask.unsqueeze(0).to(self.device)

        # Get policy
        policy, value = self.network(state, mask)
        policy = policy.squeeze(0)

        # Apply temperature
        if self.temperature != 1.0:
            policy = policy.pow(1.0 / self.temperature)
            policy = policy / policy.sum()

        # Sample action
        action_idx = torch.multinomial(policy, 1).item()

        # Convert to response
        return action_index_to_response(action_idx, decision, encoded)

    def decide_greedy(self, decision: Decision, our_player_name: str) -> str:
        """Make greedy (best) decision."""
        encoded = self.mapper.encode_decision(decision, our_player_name)

        state = encoded.state_tensor.unsqueeze(0).to(self.device)
        mask = encoded.action_mask.unsqueeze(0).to(self.device)

        policy, value = self.network(state, mask)
        action_idx = policy.argmax(dim=-1).item()

        return action_index_to_response(action_idx, decision, encoded)


if __name__ == "__main__":
    # Test state mapper
    print("Testing StateMapper...")

    mapper = StateMapper()

    # Create mock decision
    from src.forge.forge_client import GameState, PlayerState, ManaPool

    mock_player = PlayerState(
        name="TestPlayer",
        life=20,
        hand=[CardInfo(id=1, name="Lightning Bolt", cmc=1, types="Instant")],
        library_size=50,
        battlefield=[
            CardInfo(
                id=2, name="Mountain", cmc=0, types="Basic Land - Mountain",
                is_land=True, tapped=False
            )
        ],
        graveyard=[],
        exile=[],
        mana_pool=ManaPool(total=1, red=1),
    )

    mock_opponent = PlayerState(
        name="Opponent",
        life=18,
        hand=[],
        library_size=48,
        battlefield=[],
        graveyard=[],
        exile=[],
        mana_pool=ManaPool(),
    )

    mock_game_state = GameState(
        is_game_over=False,
        active_player="TestPlayer",
        priority_player="TestPlayer",
        players=[mock_player, mock_opponent],
        stack=[],
        turn=3,
        phase="MAIN1",
    )

    mock_decision = Decision(
        decision_type=DecisionType.CHOOSE_ACTION,
        decision_id=1,
        player="TestPlayer",
        turn=3,
        phase="MAIN1",
        game_state=mock_game_state,
        actions=[
            ActionOption(index=0, description="Cast Lightning Bolt", card="Lightning Bolt", card_id=1),
            ActionOption(index=-1, description="Pass", card="", card_id=-1),
        ],
    )

    encoded = mapper.encode_decision(mock_decision, "TestPlayer")

    print(f"State tensor shape: {encoded.state_tensor.shape}")
    print(f"Action mask shape: {encoded.action_mask.shape}")
    print(f"Legal actions: {encoded.legal_action_indices}")
    print(f"Descriptions: {encoded.action_descriptions}")
    print(f"Our turn: {encoded.is_our_turn}")

    print("\nStateMapper test passed!")
