#!/usr/bin/env python3
"""
Simple test script for interactive mode.
This agent plays lands and cheap creatures, then attacks.
"""

import subprocess
import json
import sys


def parse_decision(line):
    """Parse a DECISION line into a dict."""
    if not line.startswith("DECISION:"):
        return None
    try:
        return json.loads(line[9:])
    except json.JSONDecodeError:
        return None


def parse_mana_cost(cost_str):
    """Parse a mana cost string like {1}{R} into total mana needed."""
    if not cost_str:
        return 0
    total = 0
    import re
    # Match {X} patterns
    parts = re.findall(r'\{([^}]+)\}', cost_str)
    for part in parts:
        if part.isdigit():
            total += int(part)
        elif part in 'WUBRG':
            total += 1
        elif part == 'X':
            total += 1  # Assume X=1 for simplicity
    return total if total > 0 else len(parts)  # Fallback


def choose_action(decision):
    """Choose an action based on simple heuristics."""
    actions = decision.get("actions", [])
    mana_available = decision.get("mana_pool", 0)

    # First priority: play a land if we can
    for action in actions:
        if action.get("is_land", False) and action["index"] >= 0:
            return str(action["index"])

    # Second priority: play spells we can afford
    playable = []
    for action in actions:
        if action["index"] < 0:  # Skip pass option
            continue
        cost_str = action.get("mana_cost", "")
        if not cost_str:  # Free abilities
            playable.append((0, action))
        else:
            cost = parse_mana_cost(cost_str)
            if cost <= mana_available:
                playable.append((cost, action))

    if playable:
        # Play the most expensive thing we can afford
        playable.sort(key=lambda x: -x[0])
        return str(playable[0][1]["index"])

    # Default: pass
    return "-1"


def choose_attackers(decision):
    """Choose attackers - attack with everything."""
    attackers = decision.get("attackers", [])
    if not attackers:
        return ""
    # Attack with all
    return ",".join(str(a["index"]) for a in attackers)


def choose_blockers(decision):
    """Choose blockers - block the biggest attacker with smallest blocker."""
    blockers = decision.get("blockers", [])
    attackers = decision.get("attackers", [])

    if not blockers or not attackers:
        return ""

    # Find biggest attacker
    biggest_attacker = max(attackers, key=lambda x: x.get("power", 0))
    # Find smallest blocker that can survive
    best_blocker = None
    for b in blockers:
        if b.get("toughness", 0) > biggest_attacker.get("power", 0):
            if best_blocker is None or b.get("power", 0) > best_blocker.get("power", 0):
                best_blocker = b

    if best_blocker:
        return f"{best_blocker['index']}:{biggest_attacker['index']}"

    # Otherwise chump block with smallest creature
    smallest = min(blockers, key=lambda x: x.get("power", 0) + x.get("toughness", 0))
    return f"{smallest['index']}:{biggest_attacker['index']}"


def main():
    cmd = [
        "docker", "run", "--rm", "-i",
        "--entrypoint", "/bin/bash",
        "forge-sim:latest",
        "-c",
        "cd /forge && timeout 60 xvfb-run -a java -Xmx2048m "
        "--add-opens java.base/java.lang=ALL-UNNAMED "
        "--add-opens java.base/java.util=ALL-UNNAMED "
        "--add-opens java.base/java.text=ALL-UNNAMED "
        "--add-opens java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens java.desktop/java.beans=ALL-UNNAMED "
        "-Dsentry.dsn= -jar forge.jar sim "
        "-d red_aggro.dck white_weenie.dck -n 1 -i -q"
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    decision_count = 0
    max_decisions = 1000

    print("Starting interactive game...")

    try:
        while process.poll() is None and decision_count < max_decisions:
            line = process.stdout.readline()
            if not line:
                break

            line = line.strip()

            # Check for game over
            if "has won" in line.lower() or "game result" in line.lower():
                print(line)
                continue

            # Skip non-decision lines
            decision = parse_decision(line)
            if not decision:
                continue

            decision_count += 1
            dt = decision.get("decision_type", "")
            player = decision.get("player", "")
            turn = decision.get("turn", 0)
            phase = decision.get("phase", "")

            # Choose response based on decision type
            if dt == "choose_action":
                response = choose_action(decision)
                if response != "-1":
                    print(f"[Turn {turn} {phase}] {player}: Playing action {response}")
            elif dt == "declare_attackers":
                response = choose_attackers(decision)
                if response:
                    print(f"[Turn {turn}] {player}: Attacking with {response}")
            elif dt == "declare_blockers":
                response = choose_blockers(decision)
                if response:
                    print(f"[Turn {turn}] {player}: Blocking with {response}")
            else:
                response = ""

            process.stdin.write(response + "\n")
            process.stdin.flush()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
            if stdout:
                for line in stdout.split("\n"):
                    if "game result" in line.lower() or "has won" in line.lower():
                        print(line)
        except:
            process.kill()

    print(f"\nTotal decisions made: {decision_count}")


if __name__ == "__main__":
    main()
