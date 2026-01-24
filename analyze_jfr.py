#!/usr/bin/env python3
"""Analyze JFR profiling data to find hot methods."""

import subprocess
import re
from collections import defaultdict

def parse_jfr():
    """Parse JFR file and extract execution samples."""
    # Run jfr print command to get all execution samples
    result = subprocess.run(
        ["jfr", "print", "--events", "jdk.ExecutionSample", "/tmp/forge_profile.jfr"],
        capture_output=True,
        text=True
    )

    # Parse the output to extract stack traces
    samples = []
    current_sample = None
    current_stack = []
    in_stack = False

    for line in result.stdout.split('\n'):
        if line.startswith('jdk.ExecutionSample {'):
            if current_sample and current_stack:
                current_sample['stack'] = current_stack
                samples.append(current_sample)
            current_sample = {}
            current_stack = []
            in_stack = False
        elif 'sampledThread = "' in line:
            match = re.search(r'sampledThread = "([^"]+)"', line)
            if match:
                current_sample['thread'] = match.group(1)
        elif 'startTime = ' in line:
            match = re.search(r'startTime = (.+)', line)
            if match:
                current_sample['time'] = match.group(1).strip()
        elif 'stackTrace = [' in line:
            in_stack = True
        elif in_stack and line.strip().startswith(']'):
            in_stack = False
        elif in_stack and line.strip():
            # Parse stack frame
            frame = line.strip()
            if frame.endswith('...'):
                continue
            current_stack.append(frame)

    # Don't forget the last sample
    if current_sample and current_stack:
        current_sample['stack'] = current_stack
        samples.append(current_sample)

    return samples

def analyze_samples(samples):
    """Analyze samples to find hot methods."""
    # Filter to only game-related threads (exclude main thread initialization)
    game_threads = ['pool-1-thread', 'ForkJoinPool']

    method_counts = defaultdict(int)
    method_self_counts = defaultdict(int)  # Times method was at top of stack
    package_counts = defaultdict(int)

    # Filter samples to game execution (threads containing game logic)
    game_samples = []
    for s in samples:
        thread = s.get('thread', '')
        # Include samples from game threads
        if any(gt in thread for gt in game_threads) or 'pool' in thread.lower():
            game_samples.append(s)

    print(f"Total samples: {len(samples)}")
    print(f"Game-related samples: {len(game_samples)}")
    print()

    for sample in game_samples:
        stack = sample.get('stack', [])
        if not stack:
            continue

        # Count top of stack (self time)
        top_frame = stack[0]
        method_self_counts[top_frame] += 1

        # Count all methods in stack (inclusive time)
        seen = set()
        for frame in stack:
            if frame not in seen:
                method_counts[frame] += 1
                seen.add(frame)

            # Extract package
            if '.' in frame:
                parts = frame.split('.')
                if len(parts) >= 2:
                    # Get class.method
                    pkg = '.'.join(parts[:-2]) if len(parts) > 2 else parts[0]
                    package_counts[pkg] += 1

    return method_counts, method_self_counts, package_counts, len(game_samples)

def categorize_method(method):
    """Categorize a method into high-level categories."""
    method_lower = method.lower()

    if 'ai' in method_lower or 'aisimulation' in method_lower or 'spellabilitypicker' in method_lower:
        return 'AI Decision Making'
    elif 'gamecopier' in method_lower or 'copy' in method_lower:
        return 'Game State Copying'
    elif 'combat' in method_lower or 'attack' in method_lower or 'block' in method_lower:
        return 'Combat Resolution'
    elif 'trigger' in method_lower:
        return 'Trigger Processing'
    elif 'spellability' in method_lower or 'ability' in method_lower:
        return 'Ability Processing'
    elif 'cost' in method_lower or 'mana' in method_lower:
        return 'Mana/Cost Calculation'
    elif 'card' in method_lower and ('list' in method_lower or 'collection' in method_lower or 'filter' in method_lower):
        return 'Card Collection Operations'
    elif 'game' in method_lower and ('state' in method_lower or 'action' in method_lower):
        return 'Game State Management'
    elif 'java.util' in method or 'java.lang' in method:
        return 'Java Runtime'
    elif 'google' in method or 'guava' in method:
        return 'Guava Collections'
    else:
        return 'Other'

def main():
    print("=" * 70)
    print("FORGE DAEMON - JFR PROFILING ANALYSIS")
    print("=" * 70)
    print()

    print("Parsing JFR file...")
    samples = parse_jfr()

    print("Analyzing samples...")
    method_counts, method_self_counts, package_counts, total_game_samples = analyze_samples(samples)

    if total_game_samples == 0:
        print("No game-related samples found!")
        return

    print()
    print("=" * 70)
    print("TOP 30 METHODS BY SELF TIME (where CPU was actually spent)")
    print("=" * 70)

    sorted_self = sorted(method_self_counts.items(), key=lambda x: -x[1])[:30]
    for i, (method, count) in enumerate(sorted_self, 1):
        pct = count / total_game_samples * 100
        # Truncate method name for display
        display = method[:80] + "..." if len(method) > 80 else method
        print(f"{i:2}. {pct:5.1f}% ({count:4} samples) {display}")

    print()
    print("=" * 70)
    print("TOP 30 METHODS BY INCLUSIVE TIME (on stack)")
    print("=" * 70)

    sorted_inclusive = sorted(method_counts.items(), key=lambda x: -x[1])[:30]
    for i, (method, count) in enumerate(sorted_inclusive, 1):
        pct = count / total_game_samples * 100
        display = method[:80] + "..." if len(method) > 80 else method
        print(f"{i:2}. {pct:5.1f}% ({count:4} samples) {display}")

    print()
    print("=" * 70)
    print("CATEGORIZED BREAKDOWN")
    print("=" * 70)

    # Categorize methods and sum up
    category_self = defaultdict(int)
    for method, count in method_self_counts.items():
        cat = categorize_method(method)
        category_self[cat] += count

    sorted_cats = sorted(category_self.items(), key=lambda x: -x[1])
    for cat, count in sorted_cats:
        pct = count / total_game_samples * 100
        bar = '#' * int(pct / 2)
        print(f"{cat:30} {pct:5.1f}% {bar}")

    print()
    print("=" * 70)
    print("TOP FORGE PACKAGES")
    print("=" * 70)

    # Filter to forge packages only
    forge_packages = {k: v for k, v in package_counts.items() if k.startswith('forge')}
    sorted_packages = sorted(forge_packages.items(), key=lambda x: -x[1])[:20]

    for pkg, count in sorted_packages:
        pct = count / total_game_samples * 100
        print(f"{pct:5.1f}% {pkg}")

    print()
    print("=" * 70)
    print("ANALYSIS NOTES")
    print("=" * 70)
    print("""
The above shows where CPU time is spent during game execution.
Key areas to focus on for optimization:

1. Methods with high SELF TIME are where actual computation happens
2. Methods with high INCLUSIVE TIME are frequently called but may delegate work
3. AI Decision Making typically involves:
   - SpellAbilityPicker.chooseSpellAbilityToPlay()
   - AiController evaluation methods
   - GameCopier for simulating moves
4. Game State operations (CardCollection, filtering, sorting) add overhead
5. Java Runtime/Guava overhead is normal for collection-heavy code
""")

if __name__ == "__main__":
    main()
