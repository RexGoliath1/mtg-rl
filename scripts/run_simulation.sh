#!/bin/bash

# Forge MTG Headless Simulation Runner
# Usage: ./run_simulation.sh [deck1] [deck2] [options]
#
# Examples:
#   ./run_simulation.sh                           # Run default simulation
#   ./run_simulation.sh red_aggro white_weenie    # Run with specific decks
#   ./run_simulation.sh red_aggro white_weenie -n 5  # Run 5 games
#   ./run_simulation.sh red_aggro white_weenie -m 3  # Best of 3 match

set -e

# Build the Docker image if it doesn't exist
if ! docker images | grep -q "forge-mtg-sim"; then
    echo "Building Forge MTG Docker image..."
    docker build -t forge-mtg-sim .
fi

# Default arguments if none provided
if [ $# -eq 0 ]; then
    ARGS="-d red_aggro white_weenie -n 1"
else
    ARGS="$@"
fi

echo "Running Forge simulation with args: $ARGS"
echo "-------------------------------------------"

docker run --rm \
    -v "$(pwd)/decks:/forge/userdata/decks/constructed:ro" \
    forge-mtg-sim $ARGS
