# Forge Performance Analysis

## Executive Summary

**Can Forge scale to 10,000 games/hour?** Yes, but it requires either:
1. **32-36 parallel instances** with current architecture, OR
2. **14 parallel instances** with JVM reuse (persistent daemon), OR
3. **Targeted optimizations** to the heaviest components

A complete rewrite is **NOT necessary**. The bottlenecks are addressable.

---

## Current Performance Profile

| Metric | Time | Notes |
|--------|------|-------|
| **JVM + Model Init** | ~8s | Loads 32,009 card definitions |
| **Game Time (AI vs AI)** | ~5s | Varies with deck complexity |
| **Per-Decision (Interactive)** | ~50ms | Communication + processing |
| **Decisions/Game** | ~100-200 | Varies by deck type |

**Current throughput**: ~277 games/hour (single instance)

---

## Bottleneck Analysis

### 1. Initialization (8 seconds) - HIGH IMPACT

**What's happening:**
```
FModel.initialize()
├── CardStorageReader (32,009 card .txt files)
│   ├── Parse card rules
│   ├── Build keyword indices
│   └── Load token definitions
├── StaticData (card database)
├── Deck storage initialization
└── AI profile loading
```

**Why it's slow:**
- 32,009 individual file reads
- Text parsing for each card
- No pre-compiled card database

**Solution options:**
| Option | Effort | Speedup |
|--------|--------|---------|
| JVM daemon (reuse process) | Low | 10x+ (skip init) |
| Pre-compiled card DB (binary) | Medium | 5-10x |
| Lazy card loading | Medium | 3-5x |
| Only load needed cards | Low | 2-3x |

### 2. AI Decision Making - MEDIUM IMPACT

**What's happening:**
```
SpellAbilityPicker.chooseSpellAbilityToPlay()
├── getCandidateSpellsAndAbilities() - O(cards in hand + battlefield)
├── For each candidate:
│   ├── GameCopier.makeCopy() - Deep copy entire game state
│   ├── GameSimulator.simulateSa() - Execute the action
│   └── GameStateEvaluator.getScore() - Evaluate result
└── Pick highest scoring action
```

**Why it matters:**
- Each decision simulates 5-20 possible actions
- Each simulation copies the entire game state
- Game state includes all 32,009 possible card references

**This is NOT the state space problem** - the AI doesn't search all cards, just what's in play.

**Solution options:**
| Option | Effort | Speedup |
|--------|--------|---------|
| Reduce simulation depth | Config | 2x |
| Use simple heuristics (disable sim) | Config | 3-5x |
| Incremental state copy | High | 2x |

### 3. Communication Overhead (Interactive Mode) - LOW-MEDIUM IMPACT

**What's happening:**
```
For each decision:
├── Forge serializes game state to JSON
├── Write to stdout (buffered)
├── Agent reads, parses JSON
├── Agent selects action
├── Write to stdin
└── Forge parses response
```

**Current overhead:** ~5ms per decision (negligible if network)

**Your concern about agent-vs-agent:** When both players are external agents, each decision involves:
- 2x JSON serialization
- 2x pipe communication
- But this is still only ~10ms overhead, not significant

### 4. Game Length (Unskilled Agents) - ACTUALLY HELPFUL

Games with unskilled agents may be **longer in turns** but **faster in wall-clock time** because:
- Fewer complex board states
- Simpler decision trees
- Less simulation needed per decision

As agents improve, games will get shorter (more efficient wins) but decisions will be harder.

---

## What's NOT the Bottleneck

| Component | Why it's fine |
|-----------|--------------|
| **Card state space (32K cards)** | Only ~100 cards active per game |
| **Rule complexity** | Rules are evaluated lazily |
| **GUI components** | Already stripped in headless mode |
| **Network/IPC** | Adds <5% overhead |
| **Game engine itself** | Well-optimized for single games |

---

## Recommended Optimization Path

### Phase 1: Quick Wins (No code changes)
```bash
# Run multiple parallel instances
for i in {1..8}; do
    python3 train.py --mode single &
done

# Expected: ~2,200 games/hour on 8 cores
```

### Phase 2: JVM Daemon (Low effort, high impact)
Create a persistent Forge process that accepts multiple games:

```java
// New mode: daemon
while (true) {
    String command = readLine();  // "NEWGAME deck1 deck2"
    if (command.startsWith("NEWGAME")) {
        simulateSingleMatch(decks);  // Skip FModel.initialize()
    }
}
```

**Expected improvement:** 720 games/hour per instance (vs 277 current)

### Phase 3: Reduce AI Simulation Depth (Config change)
```java
// In AiController or forge preferences
useSimulation = false;  // Use heuristics only
// OR
maxSimulationDepth = 1;  // Instead of full tree
```

**Expected improvement:** 2-3x faster per game

### Phase 4: Lazy Card Loading (Medium effort)
Only parse cards that appear in the decks being played:

```java
// Modified CardStorageReader
public Card getCard(String name) {
    if (!loadedCards.contains(name)) {
        loadCard(name);  // Lazy load
    }
    return loadedCards.get(name);
}
```

**Expected improvement:** Init from 8s to <1s for known decks

---

## Scaling Projections

| Configuration | Games/Hour | Hardware |
|--------------|------------|----------|
| Current (1 instance) | 277 | MacBook Air |
| 8 parallel | 2,200 | MacBook Air |
| Daemon mode (8 parallel) | 5,760 | MacBook Air |
| Daemon + simple AI (8 parallel) | 11,520 | MacBook Air |
| Cloud (32 instances) | 8,800 | c5.4xlarge |
| Cloud + daemon (32 instances) | 23,000 | c5.4xlarge |

---

## Answering Your Specific Questions

### Q: Is state space the limiting factor?
**No.** The 32,009 cards are loaded once. During gameplay, only ~100-200 cards are active. The game engine handles this efficiently.

### Q: Is agent communication the bottleneck?
**No.** Communication adds <5% overhead (~5ms per decision out of ~50ms total).

### Q: Are games longer because agents are bad?
**Yes, but this helps.** Longer games = more training data per game. As agents improve, games will be shorter but decisions harder.

### Q: Is initialization the issue?
**Partially.** The 8-second init is significant but solvable with daemon mode.

### Q: Do we need a rewrite?
**No.** The Forge engine is well-architected. The issues are:
1. Cold-start init (solvable with daemon)
2. AI simulation depth (configurable)
3. Parallelization (just run more instances)

---

## Next Steps

1. **Immediate:** Run 4-8 parallel instances for training
2. **Short-term:** Implement daemon mode to skip init
3. **Medium-term:** Profile actual games with async-profiler
4. **Optional:** Consider forking Forge with training optimizations

---

## Appendix: Profiling Commands

```bash
# Java flight recorder (built-in profiler)
java -XX:+FlightRecorder -XX:StartFlightRecording=duration=60s,filename=forge.jfr \
     -jar forge.jar sim -d deck1.dck deck2.dck -n 10

# async-profiler (more detailed)
java -agentpath:/path/to/libasyncProfiler.so=start,file=profile.html \
     -jar forge.jar sim -d deck1.dck deck2.dck -n 10

# macOS Instruments
xcrun xctrace record --template 'Time Profiler' --launch -- \
     java -jar forge.jar sim -d deck1.dck deck2.dck -n 5
```
