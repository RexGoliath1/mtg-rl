"""
Monte Carlo Tree Search (MCTS) for MTG

Implementation of AlphaZero-style MCTS for Magic: The Gathering gameplay.
Uses the policy network to guide exploration and value network for evaluation.

Key Differences from Chess/Go MCTS:
1. Hidden Information: Sample opponent hands for imperfect info
2. Variable Actions: Dynamic legal moves (1-100+) each state
3. Stack Resolution: LIFO stack requires careful state tracking
4. Multiplayer: Commander has 4 players (addressed separately)

Algorithm:
1. SELECT: Traverse tree using PUCT formula (exploration vs exploitation)
2. EXPAND: Add new node using policy network prior
3. EVALUATE: Use value network (no rollout needed)
4. BACKUP: Update visit counts and values up the tree

After N simulations, select action based on visit counts.

Usage:
    mcts = MCTS(policy_value_net, forge_client)
    action_probs = mcts.search(game_state, num_simulations=800)
    action = select_action(action_probs, temperature=1.0)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MCTSConfig:
    """Configuration for MCTS search."""

    # Search parameters
    num_simulations: int = 800    # Number of MCTS simulations per move
    c_puct: float = 1.5            # Exploration constant in PUCT formula
    dirichlet_alpha: float = 0.3   # Dirichlet noise alpha for root
    dirichlet_epsilon: float = 0.25  # Fraction of prior to replace with noise

    # Tree policy
    temperature: float = 1.0       # Action selection temperature
    temperature_drop_move: int = 30  # Move to drop temperature to 0 (deterministic)

    # Value head
    value_discount: float = 0.99   # Discount factor for future rewards

    # Practical limits
    max_depth: int = 200           # Maximum search depth
    max_tree_size: int = 100000    # Maximum nodes in tree

    # Hidden information handling
    num_info_samples: int = 10     # Number of opponent hand samples
    determinization: str = "single"  # "single" or "ensemble"

    # Device
    device: str = "cpu"


# =============================================================================
# MCTS NODE
# =============================================================================

class MCTSNode:
    """
    A node in the MCTS tree.

    Each node represents a game state after taking an action.
    Stores statistics for the tree policy (PUCT).
    """

    def __init__(
        self,
        prior: float = 0.0,
        parent: Optional['MCTSNode'] = None,
        action: Optional[int] = None,
    ):
        """
        Create a new MCTS node.

        Args:
            prior: Prior probability from policy network P(a|s)
            parent: Parent node in the tree
            action: Action taken to reach this node
        """
        self.prior = prior
        self.parent = parent
        self.action = action

        # Statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, MCTSNode] = {}

        # State info (populated when expanded)
        self.game_state: Optional[Dict[str, Any]] = None
        self.legal_actions: Optional[List[int]] = None
        self.is_terminal = False
        self.terminal_value: Optional[float] = None

    @property
    def value(self) -> float:
        """Average value of this node (Q-value)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        """Whether this node has been expanded."""
        return len(self.children) > 0

    def expand(
        self,
        action_priors: np.ndarray,
        legal_actions: List[int],
        game_state: Dict[str, Any],
    ):
        """
        Expand this node with children for each legal action.

        Args:
            action_priors: Prior probabilities from policy network
            legal_actions: List of legal action indices
            game_state: Current game state
        """
        self.game_state = game_state
        self.legal_actions = legal_actions

        for action in legal_actions:
            self.children[action] = MCTSNode(
                prior=action_priors[action],
                parent=self,
                action=action,
            )

    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        """
        Select best child using PUCT formula.

        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

        Args:
            c_puct: Exploration constant

        Returns:
            (action, child_node) with highest PUCT value
        """
        total_visits = sum(child.visit_count for child in self.children.values())
        sqrt_total = math.sqrt(total_visits + 1)

        best_score = float('-inf')
        best_action = None
        best_child = None

        for action, child in self.children.items():
            # Exploitation: average value
            exploitation = child.value

            # Exploration: prior * sqrt(parent visits) / (1 + child visits)
            exploration = c_puct * child.prior * sqrt_total / (1 + child.visit_count)

            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def backup(self, value: float):
        """
        Backup value through the tree.

        Updates visit count and value sum for this node and all ancestors.

        Args:
            value: Value to backup (from perspective of player who moved)
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            # Flip value for opponent's perspective
            value = -value
            node = node.parent

    def get_action_probs(self, temperature: float = 1.0) -> np.ndarray:
        """
        Get action probabilities based on visit counts.

        Args:
            temperature: Temperature for action selection
                - 0: Deterministic (max visit count)
                - 1: Proportional to visit counts
                - >1: More uniform

        Returns:
            Action probabilities (sparse, only legal actions non-zero)
        """
        if not self.is_expanded:
            return np.zeros(1)

        visits = np.array([
            self.children[a].visit_count if a in self.children else 0
            for a in range(max(self.children.keys()) + 1)
        ])

        if temperature == 0:
            # Deterministic: select max
            probs = np.zeros_like(visits, dtype=np.float32)
            probs[np.argmax(visits)] = 1.0
        else:
            # Temperature-scaled probabilities
            visits = visits ** (1.0 / temperature)
            total = visits.sum()
            if total > 0:
                probs = visits / total
            else:
                probs = np.ones_like(visits, dtype=np.float32) / len(visits)

        return probs


# =============================================================================
# FORGE CLIENT INTERFACE
# =============================================================================

class ForgeClientInterface:
    """
    Interface for communicating with Forge game engine.

    This is an abstract interface - implement for your Forge setup.
    """

    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state as JSON dict."""
        raise NotImplementedError

    def get_legal_actions(self) -> List[Dict[str, Any]]:
        """Get list of legal actions."""
        raise NotImplementedError

    def apply_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply action and return new game state.

        Args:
            action: Action dict (type, index, parameters)

        Returns:
            New game state after action
        """
        raise NotImplementedError

    def clone_state(self) -> 'ForgeClientInterface':
        """Create a copy of current game state for simulation."""
        raise NotImplementedError

    def is_game_over(self) -> bool:
        """Check if game has ended."""
        raise NotImplementedError

    def get_game_result(self, player_idx: int) -> float:
        """
        Get game result for a player.

        Args:
            player_idx: Player index

        Returns:
            1.0 for win, -1.0 for loss, 0.0 for draw
        """
        raise NotImplementedError


class SimulatedForgeClient(ForgeClientInterface):
    """
    Simulated Forge client for testing.

    Uses a simple game model instead of actual Forge.
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self.state = initial_state or self._create_initial_state()
        self.move_count = 0

    def _create_initial_state(self) -> Dict[str, Any]:
        return {
            "turn": 1,
            "phase": "main1",
            "activePlayer": 0,
            "priorityPlayer": 0,
            "players": [
                {"id": 0, "life": 20, "hand": [], "battlefield": []},
                {"id": 1, "life": 20, "hand": [], "battlefield": []},
            ],
            "stack": [],
            "gameOver": False,
            "winner": None,
        }

    def get_game_state(self) -> Dict[str, Any]:
        return self.state.copy()

    def get_legal_actions(self) -> List[Dict[str, Any]]:
        # Always have pass available
        actions = [{"type": "pass", "index": 0}]

        # Add some random legal actions
        if self.move_count < 50:
            if len(self.state["players"][0]["hand"]) > 0:
                actions.append({"type": "cast", "index": 0})
            if len(self.state["players"][0]["battlefield"]) > 0:
                actions.append({"type": "activate", "index": 0})

        return actions

    def apply_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self.move_count += 1

        # Simulate game progression
        if action["type"] == "pass":
            # Progress phase/turn
            if self.state["phase"] == "main1":
                self.state["phase"] = "combat"
            elif self.state["phase"] == "combat":
                self.state["phase"] = "main2"
            elif self.state["phase"] == "main2":
                self.state["phase"] = "end"
            else:
                self.state["phase"] = "main1"
                self.state["turn"] += 1

                # Swap active player
                self.state["activePlayer"] = 1 - self.state["activePlayer"]

        # Random game end condition
        if self.move_count > 100 or self.state["turn"] > 20:
            self.state["gameOver"] = True
            self.state["winner"] = 0 if np.random.random() > 0.5 else 1

        return self.get_game_state()

    def clone_state(self) -> 'SimulatedForgeClient':
        import copy
        clone = SimulatedForgeClient()
        clone.state = copy.deepcopy(self.state)
        clone.move_count = self.move_count
        return clone

    def is_game_over(self) -> bool:
        return self.state.get("gameOver", False)

    def get_game_result(self, player_idx: int) -> float:
        winner = self.state.get("winner")
        if winner is None:
            return 0.0
        return 1.0 if winner == player_idx else -1.0


# =============================================================================
# MCTS
# =============================================================================

class MCTS:
    """
    Monte Carlo Tree Search for MTG.

    Uses neural network for policy (exploration guidance) and value (evaluation).
    Communicates with Forge for game simulation.
    """

    def __init__(
        self,
        policy_value_fn: Callable,
        encode_state_fn: Callable,
        config: Optional[MCTSConfig] = None,
    ):
        """
        Initialize MCTS.

        Args:
            policy_value_fn: Function that takes state tensor, returns (policy, value)
                            Policy: [action_dim] probabilities
                            Value: scalar in [-1, 1]
            encode_state_fn: Function that takes game state dict, returns tensor
            config: MCTS configuration
        """
        self.policy_value_fn = policy_value_fn
        self.encode_state_fn = encode_state_fn
        self.config = config or MCTSConfig()

        self.root: Optional[MCTSNode] = None
        self.current_player = 0

    def search(
        self,
        forge_client: ForgeClientInterface,
        action_mask: np.ndarray,
        num_simulations: Optional[int] = None,
    ) -> np.ndarray:
        """
        Run MCTS search from current game state.

        Args:
            forge_client: Interface to Forge game engine
            action_mask: [action_dim] binary mask of legal actions
            num_simulations: Number of simulations (default from config)

        Returns:
            action_probs: [action_dim] probabilities based on visit counts
        """
        num_simulations = num_simulations or self.config.num_simulations

        # Get initial state
        game_state = forge_client.get_game_state()
        self.current_player = game_state.get("priorityPlayer", 0)

        # Create root if needed
        if self.root is None:
            self.root = MCTSNode()

        # Expand root if needed
        if not self.root.is_expanded:
            self._expand_node(self.root, game_state, action_mask)

        # Add Dirichlet noise at root for exploration
        self._add_dirichlet_noise(self.root)

        # Run simulations
        for _ in range(num_simulations):
            # Clone game state for simulation
            sim_client = forge_client.clone_state()

            # Simulate from root
            self._simulate(self.root, sim_client)

        # Get action probabilities from visit counts
        return self.root.get_action_probs(self.config.temperature)

    def _simulate(self, node: MCTSNode, forge_client: ForgeClientInterface):
        """
        Run one MCTS simulation (select, expand, evaluate, backup).
        """
        # SELECT: Traverse tree to leaf
        path = [node]
        current_node = node

        while current_node.is_expanded and not current_node.is_terminal:
            action, current_node = current_node.select_child(self.config.c_puct)
            path.append(current_node)

            # Apply action in simulation
            action_dict = self._decode_action(action)
            forge_client.apply_action(action_dict)

        # Check for terminal state
        if forge_client.is_game_over():
            current_node.is_terminal = True
            current_node.terminal_value = forge_client.get_game_result(self.current_player)
            value = current_node.terminal_value
        else:
            # EXPAND: Add children for this node
            game_state = forge_client.get_game_state()
            legal_actions = forge_client.get_legal_actions()
            action_mask = self._create_action_mask(legal_actions)

            if not current_node.is_expanded:
                self._expand_node(current_node, game_state, action_mask)

            # EVALUATE: Get value from neural network
            value = self._evaluate(game_state)

        # BACKUP: Update values along path
        for i, n in enumerate(reversed(path)):
            # Alternate sign for opponent's perspective
            n.visit_count += 1
            if i % 2 == 0:
                n.value_sum += value
            else:
                n.value_sum -= value

    def _expand_node(
        self,
        node: MCTSNode,
        game_state: Dict[str, Any],
        action_mask: np.ndarray
    ):
        """Expand a node with children for each legal action."""
        # Get policy prior from network
        state_tensor = self.encode_state_fn(game_state)
        with torch.no_grad():
            policy, _ = self.policy_value_fn(state_tensor)

        # Apply mask and normalize
        policy = policy.cpu().numpy().flatten()
        policy = policy * action_mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # Uniform over legal actions
            policy = action_mask / action_mask.sum()

        # Find legal action indices
        legal_actions = np.nonzero(action_mask)[0].tolist()

        # Expand node
        node.expand(policy, legal_actions, game_state)

    def _evaluate(self, game_state: Dict[str, Any]) -> float:
        """Get value estimate from neural network."""
        state_tensor = self.encode_state_fn(game_state)
        with torch.no_grad():
            _, value = self.policy_value_fn(state_tensor)
        return value.item()

    def _add_dirichlet_noise(self, node: MCTSNode):
        """Add Dirichlet noise to root for exploration."""
        if not node.is_expanded:
            return

        num_actions = len(node.children)
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * num_actions)

        eps = self.config.dirichlet_epsilon
        for i, (action, child) in enumerate(node.children.items()):
            child.prior = (1 - eps) * child.prior + eps * noise[i]

    def _create_action_mask(self, legal_actions: List[Dict[str, Any]]) -> np.ndarray:
        """Convert Forge legal actions to action mask."""
        from src.forge.policy_value_heads import create_action_mask, ActionConfig
        return create_action_mask(legal_actions, ActionConfig())

    def _decode_action(self, action_idx: int) -> Dict[str, Any]:
        """Convert action index to Forge action dict."""
        from src.forge.policy_value_heads import decode_action
        return decode_action(action_idx)

    def update_with_action(self, action: int):
        """
        Update tree after action is taken.

        Reuses subtree rooted at chosen action.

        Args:
            action: Action index that was taken
        """
        if self.root is not None and action in self.root.children:
            # Reuse subtree
            new_root = self.root.children[action]
            new_root.parent = None
            self.root = new_root
        else:
            # Reset tree
            self.root = None

    def reset(self):
        """Reset the search tree."""
        self.root = None


# =============================================================================
# PARALLEL MCTS (for batched evaluation)
# =============================================================================

class BatchedMCTS:
    """
    Batched MCTS for parallel game simulation.

    Runs multiple games in parallel, batching neural network evaluations
    for efficient GPU utilization.
    """

    def __init__(
        self,
        policy_value_fn: Callable,
        encode_state_fn: Callable,
        num_parallel: int = 8,
        config: Optional[MCTSConfig] = None,
    ):
        """
        Initialize batched MCTS.

        Args:
            policy_value_fn: Batched policy-value function
            encode_state_fn: State encoding function
            num_parallel: Number of parallel games
            config: MCTS configuration
        """
        self.policy_value_fn = policy_value_fn
        self.encode_state_fn = encode_state_fn
        self.num_parallel = num_parallel
        self.config = config or MCTSConfig()

        # One MCTS instance per parallel game
        self.mcts_instances = [
            MCTS(policy_value_fn, encode_state_fn, config)
            for _ in range(num_parallel)
        ]

    def search_batch(
        self,
        forge_clients: List[ForgeClientInterface],
        action_masks: np.ndarray,
    ) -> np.ndarray:
        """
        Run MCTS search for multiple games in parallel.

        Args:
            forge_clients: List of Forge client instances
            action_masks: [batch, action_dim] binary masks

        Returns:
            action_probs: [batch, action_dim] probabilities
        """
        assert len(forge_clients) == self.num_parallel
        assert len(action_masks) == self.num_parallel

        # Run search for each game
        # TODO: Implement proper batching for neural network calls
        results = []
        for i, (mcts, client, mask) in enumerate(
            zip(self.mcts_instances, forge_clients, action_masks)
        ):
            probs = mcts.search(client, mask)
            results.append(probs)

        # Pad to same size
        max_len = max(len(p) for p in results)
        padded = np.zeros((len(results), max_len))
        for i, p in enumerate(results):
            padded[i, :len(p)] = p

        return padded


# =============================================================================
# TESTING
# =============================================================================

def test_mcts():
    """Test MCTS implementation."""
    print("=" * 70)
    print("Testing MCTS")
    print("=" * 70)

    # Create mock policy-value function
    def mock_policy_value(state_tensor):
        """Mock network that returns uniform policy and neutral value."""
        batch_size = state_tensor.shape[0] if len(state_tensor.shape) > 1 else 1
        policy = torch.softmax(torch.randn(batch_size, 153), dim=-1)
        value = torch.zeros(batch_size, 1)
        return policy, value

    # Create mock state encoder
    def mock_encode_state(game_state):
        """Mock encoder that returns random tensor."""
        return torch.randn(1, 512)

    # Create MCTS
    config = MCTSConfig(num_simulations=50)  # Fewer sims for testing
    mcts = MCTS(mock_policy_value, mock_encode_state, config)

    # Create simulated Forge client
    forge_client = SimulatedForgeClient()

    # Create action mask
    legal_actions = forge_client.get_legal_actions()
    from src.forge.policy_value_heads import create_action_mask
    action_mask = create_action_mask(legal_actions)

    print(f"\nInitial state: {forge_client.get_game_state()}")
    print(f"Legal actions: {legal_actions}")
    print(f"Action mask sum: {action_mask.sum()}")

    # Run search
    print(f"\nRunning {config.num_simulations} MCTS simulations...")
    action_probs = mcts.search(forge_client, action_mask, num_simulations=50)

    print(f"\nAction probabilities shape: {action_probs.shape}")
    print(f"Non-zero probs: {np.nonzero(action_probs)[0]}")
    for idx in np.nonzero(action_probs)[0]:
        print(f"  Action {idx}: prob={action_probs[idx]:.4f}")

    # Check root statistics
    print(f"\nRoot node statistics:")
    print(f"  Visit count: {mcts.root.visit_count}")
    print(f"  Value: {mcts.root.value:.4f}")
    print(f"  Children: {len(mcts.root.children)}")

    # Select action and update tree
    selected_action = np.argmax(action_probs)
    print(f"\nSelected action: {selected_action}")

    mcts.update_with_action(selected_action)
    print(f"After update - root visit count: {mcts.root.visit_count if mcts.root else 'None'}")

    print("\n" + "=" * 70)
    print("MCTS test completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_mcts()
