#!/usr/bin/env python3
"""
Agent Wrapper for Forge MTG Simulation

This module provides a Python interface for communicating with the Forge game engine
in interactive mode. It handles spawning the game process, parsing JSON decision
requests, and sending responses back.

Usage:
    from agent_wrapper import ForgeGame, RandomAgent

    agent1 = RandomAgent("Agent1")
    agent2 = RandomAgent("Agent2")

    game = ForgeGame(
        deck1="red_aggro.dck",
        deck2="white_weenie.dck"
    )

    result = game.play(agent1, agent2)
"""

import json
import subprocess
import sys
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import threading
import queue


@dataclass
class GameState:
    """Represents the current game state."""
    turn: int
    phase: str
    active_player: str
    players: List[Dict[str, Any]]
    battlefield: List[Dict[str, Any]]
    stack: List[Dict[str, Any]]


@dataclass
class Decision:
    """Represents a decision request from the game engine."""
    decision_id: int
    decision_type: str
    player: str
    turn: int
    phase: str
    data: Dict[str, Any]

    @classmethod
    def from_json(cls, json_str: str) -> 'Decision':
        """Parse a decision from JSON string."""
        data = json.loads(json_str)
        return cls(
            decision_id=data.get('decision_id', 0),
            decision_type=data.get('decision_type', ''),
            player=data.get('player', ''),
            turn=data.get('turn', 0),
            phase=data.get('phase', ''),
            data=data
        )


class Agent(ABC):
    """Abstract base class for game agents."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def decide(self, decision: Decision, game_state: Optional[GameState]) -> str:
        """Make a decision based on the current game state and decision request.

        Args:
            decision: The decision request from the game engine
            game_state: Current game state (may be None if not available)

        Returns:
            String response to send back to the game engine
        """
        pass


class RandomAgent(Agent):
    """An agent that makes random decisions."""

    def decide(self, decision: Decision, game_state: Optional[GameState]) -> str:
        """Make a random decision."""
        dt = decision.decision_type
        data = decision.data

        if dt == "declare_attackers":
            # Randomly choose some attackers
            attackers = data.get("attackers", [])
            if not attackers:
                return ""
            # Randomly select 0-all attackers
            num_to_attack = random.randint(0, len(attackers))
            selected = random.sample(range(len(attackers)), num_to_attack)
            return ",".join(str(i) for i in selected)

        elif dt == "declare_blockers":
            # Randomly assign blockers to attackers
            blockers = data.get("blockers", [])
            attackers = data.get("attackers", [])
            if not blockers or not attackers:
                return ""
            # Randomly assign some blockers
            assignments = []
            for i, blocker in enumerate(blockers):
                if random.random() > 0.5:
                    attacker_idx = random.randint(0, len(attackers) - 1)
                    assignments.append(f"{i}:{attacker_idx}")
            return ",".join(assignments)

        elif dt == "choose_ability":
            # Choose random ability
            abilities = data.get("abilities", [])
            if not abilities:
                return "0"
            return str(random.randint(0, len(abilities) - 1))

        elif dt == "choose_cards":
            # Choose random cards within limits
            cards = data.get("cards", [])
            min_cards = data.get("min", 0)
            max_cards = data.get("max", len(cards))
            if not cards:
                return ""
            num_to_choose = random.randint(min_cards, min(max_cards, len(cards)))
            selected = random.sample(range(len(cards)), num_to_choose)
            return ",".join(str(i) for i in selected)

        elif dt == "choose_entity":
            # Choose random entity
            entities = data.get("entities", [])
            optional = data.get("optional", False)
            if not entities:
                return "-1" if optional else "0"
            if optional and random.random() < 0.1:
                return "-1"
            return str(random.randint(0, len(entities) - 1))

        elif dt == "confirm_action":
            # Random yes/no
            return "y" if random.random() > 0.5 else "n"

        elif dt == "play_trigger":
            # Usually play optional triggers
            return "y" if random.random() > 0.3 else "n"

        elif dt == "play_from_effect":
            # Usually play from effect
            return "y" if random.random() > 0.2 else "n"

        elif dt == "announce_value":
            # Announce a reasonable value
            return "1"

        elif dt == "reveal":
            # Just acknowledge reveal
            return ""

        else:
            # Default: return empty or first option
            return ""


class PassAgent(Agent):
    """An agent that always passes/declines when possible."""

    def decide(self, decision: Decision, game_state: Optional[GameState]) -> str:
        """Always pass or choose minimal/no action."""
        dt = decision.decision_type

        if dt == "declare_attackers":
            return ""  # Don't attack
        elif dt == "declare_blockers":
            return ""  # Don't block
        elif dt == "confirm_action":
            return "n"  # Decline
        elif dt == "play_trigger":
            return "n"  # Don't play optional triggers
        elif dt == "play_from_effect":
            return "n"  # Don't play from effect
        elif dt == "choose_cards":
            data = decision.data
            if data.get("optional", False) or data.get("min", 0) == 0:
                return ""  # Choose nothing
            return "0"  # Choose minimum
        else:
            return ""


class ForgeGame:
    """Manages a game session with the Forge engine."""

    def __init__(
        self,
        deck1: str,
        deck2: str,
        docker_image: str = "forge-sim:latest",
        game_format: str = "Constructed",
        timeout: int = 120
    ):
        """Initialize a Forge game session.

        Args:
            deck1: Path to first deck file
            deck2: Path to second deck file
            docker_image: Docker image to use
            game_format: Game format (Constructed, Commander, etc.)
            timeout: Timeout in seconds for each game
        """
        self.deck1 = deck1
        self.deck2 = deck2
        self.docker_image = docker_image
        self.game_format = game_format
        self.timeout = timeout
        self.process: Optional[subprocess.Popen] = None
        self.game_state: Optional[GameState] = None

    def play(self, agent1: Agent, agent2: Agent) -> Dict[str, Any]:
        """Play a game between two agents.

        Args:
            agent1: Agent controlling player 1
            agent2: Agent controlling player 2

        Returns:
            Dictionary containing game result
        """
        # Build the command
        cmd = [
            "docker", "run", "--rm", "-i",
            "--entrypoint", "/bin/bash",
            self.docker_image,
            "-c",
            f"cd /forge && xvfb-run -a java -Xmx2048m "
            f"--add-opens java.base/java.lang=ALL-UNNAMED "
            f"--add-opens java.base/java.util=ALL-UNNAMED "
            f"--add-opens java.base/java.text=ALL-UNNAMED "
            f"--add-opens java.base/java.lang.reflect=ALL-UNNAMED "
            f"--add-opens java.desktop/java.beans=ALL-UNNAMED "
            f"-Dsentry.dsn= -jar forge.jar sim "
            f"-d {self.deck1} {self.deck2} -n 1 -i -c {self.timeout}"
        ]

        # Start the process
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        result = {
            "winner": None,
            "turns": 0,
            "decisions": [],
            "error": None
        }

        agents = {
            agent1.name: agent1,
            agent2.name: agent2
        }

        # Map player names to agents (will be set when we see first decision)
        player_to_agent: Dict[str, Agent] = {}

        try:
            while self.process.poll() is None:
                line = self.process.stdout.readline()
                if not line:
                    continue

                line = line.strip()

                # Skip non-decision lines
                if not line.startswith("DECISION:"):
                    # Check for game results
                    if "has won" in line.lower():
                        # Parse winner
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p.lower() == "has" and i > 0:
                                result["winner"] = parts[i-1]
                                break
                    elif "turn" in line.lower() and ":" in line:
                        # Parse turn number
                        try:
                            turn_str = line.split("Turn")[1].split()[0]
                            result["turns"] = int(turn_str)
                        except (IndexError, ValueError):
                            pass
                    continue

                # Parse decision
                json_str = line[9:]  # Remove "DECISION:" prefix
                try:
                    decision = Decision.from_json(json_str)
                except json.JSONDecodeError:
                    continue

                # Map player to agent (first time we see a player)
                player_name = decision.player
                if player_name not in player_to_agent:
                    # Assign agents to players in order
                    if len(player_to_agent) == 0:
                        player_to_agent[player_name] = agent1
                    else:
                        player_to_agent[player_name] = agent2

                # Get the agent for this player
                agent = player_to_agent.get(player_name)
                if not agent:
                    response = ""
                else:
                    response = agent.decide(decision, self.game_state)

                # Record decision
                result["decisions"].append({
                    "decision_id": decision.decision_id,
                    "type": decision.decision_type,
                    "player": player_name,
                    "response": response
                })

                # Send response
                self.process.stdin.write(response + "\n")
                self.process.stdin.flush()

        except Exception as e:
            result["error"] = str(e)
        finally:
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()

        return result


def main():
    """Main entry point for testing."""
    print("Testing Forge Agent Wrapper")
    print("-" * 40)

    # Create agents
    agent1 = RandomAgent("RandomAgent1")
    agent2 = RandomAgent("RandomAgent2")

    # Create game
    game = ForgeGame(
        deck1="red_aggro.dck",
        deck2="white_weenie.dck",
        timeout=60
    )

    print(f"Starting game: {game.deck1} vs {game.deck2}")
    print("Using random agents for both players")
    print("-" * 40)

    # Play the game
    result = game.play(agent1, agent2)

    # Print results
    print(f"Game ended after {result['turns']} turns")
    print(f"Winner: {result['winner']}")
    print(f"Total decisions made: {len(result['decisions'])}")

    if result['error']:
        print(f"Error: {result['error']}")

    # Show decision summary
    decision_types = {}
    for d in result['decisions']:
        dt = d['type']
        decision_types[dt] = decision_types.get(dt, 0) + 1

    print("\nDecision type breakdown:")
    for dt, count in sorted(decision_types.items()):
        print(f"  {dt}: {count}")


if __name__ == "__main__":
    main()
