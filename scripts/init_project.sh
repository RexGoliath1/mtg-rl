#!/usr/bin/env bash
# init_project.sh — Bootstrap Claude Code memory, testing, CI, and dev tooling for a new project.
#
# Usage:
#   ./init_project.sh "My Project"                    # Python project (default)
#   ./init_project.sh "My Project" --node              # Node.js project
#   ./init_project.sh "My Project" --minimal           # Just CLAUDE.md, memory, .gitignore
#   ./init_project.sh "My Project" --python --node     # Both ecosystems
#
# Idempotent: safe to re-run. Existing files are never overwritten.

set -eo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

created_files=()

log_created() {
    created_files+=("$1")
    printf "${GREEN}  + %s${NC}\n" "$1"
}

log_skipped() {
    printf "${YELLOW}  ~ %s (already exists)${NC}\n" "$1"
}

write_if_missing() {
    local filepath="$1"
    local content="$2"
    local dir
    dir="$(dirname "$filepath")"

    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
    fi

    if [ -f "$filepath" ]; then
        log_skipped "$filepath"
        return 0
    fi

    printf '%s\n' "$content" > "$filepath"
    log_created "$filepath"
    return 0
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

PROJECT_NAME=""
FLAG_PYTHON=false
FLAG_NODE=false
FLAG_MINIMAL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)  FLAG_PYTHON=true; shift ;;
        --node)    FLAG_NODE=true; shift ;;
        --minimal) FLAG_MINIMAL=true; shift ;;
        --help|-h)
            printf "Usage: %s \"Project Name\" [--python] [--node] [--minimal]\n" "$0"
            printf "\n  --python   Python project with UV/pyproject.toml (default)\n"
            printf "  --node     Node.js project with package.json\n"
            printf "  --minimal  Just CLAUDE.md, memory dir, and .gitignore\n"
            exit 0
            ;;
        -*)
            printf "Unknown flag: %s\n" "$1" >&2
            exit 1
            ;;
        *)
            if [ -z "$PROJECT_NAME" ]; then
                PROJECT_NAME="$1"
            else
                printf "Unexpected argument: %s\n" "$1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$PROJECT_NAME" ]; then
    printf "Error: Project name is required.\nUsage: %s \"Project Name\" [--python] [--node] [--minimal]\n" "$0" >&2
    exit 1
fi

# Default to --python if no ecosystem flag was given and not --minimal
if ! $FLAG_PYTHON && ! $FLAG_NODE && ! $FLAG_MINIMAL; then
    FLAG_PYTHON=true
fi

# Derive package name: lowercase, replace spaces/hyphens with underscores, strip non-alnum
PKG_NAME=$(printf '%s' "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]' | tr ' -' '_' | tr -cd 'a-z0-9_')

PROJECT_DIR="$(pwd)"

printf "\n${CYAN}Initializing project: %s${NC}\n" "$PROJECT_NAME"
printf "  Directory: %s\n" "$PROJECT_DIR"
printf "  Package:   %s\n" "$PKG_NAME"
printf "  Flags:     python=%s node=%s minimal=%s\n\n" "$FLAG_PYTHON" "$FLAG_NODE" "$FLAG_MINIMAL"

# ---------------------------------------------------------------------------
# 1. Git init
# ---------------------------------------------------------------------------

if [ ! -d .git ]; then
    git init
    printf "${GREEN}  + Initialized git repository${NC}\n"
else
    printf "${YELLOW}  ~ Git repo already initialized${NC}\n"
fi

# ---------------------------------------------------------------------------
# 2. .gitignore
# ---------------------------------------------------------------------------

GITIGNORE_CONTENT="# === OS ===
.DS_Store
Thumbs.db
*.swp
*~

# === IDE ===
.vscode/
.idea/
*.code-workspace

# === Environment ===
.env
.env.*
!.env.example

# === Python ===
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
*.egg
.venv/
venv/

# === Node.js ===
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# === Data & Models ===
data/
*.csv
*.csv.gz
*.parquet
*.jsonl
*.h5
*.hdf5

# === Model weights ===
*.pt
*.pth
*.onnx
*.safetensors
checkpoints/

# === Logs ===
logs/
*.log
wandb/
runs/

# === Coverage ===
htmlcov/
.coverage
coverage.xml

# === Misc ===
*.bak
*.tmp"

write_if_missing ".gitignore" "$GITIGNORE_CONTENT"

# ---------------------------------------------------------------------------
# 3. Project CLAUDE.md
# ---------------------------------------------------------------------------

CLAUDE_MD="# CLAUDE.md - ${PROJECT_NAME}

## Project Overview

**${PROJECT_NAME}** - [Brief description of the project].

**Goal**: [What this project aims to accomplish].

---

## Quick Start

\`\`\`bash
# Check project status
git status
git log --oneline -5
"

if $FLAG_PYTHON || $FLAG_MINIMAL; then
    CLAUDE_MD+="
# Install dependencies
uv sync --extra dev

# Run tests
uv run python3 -m pytest tests/ -v

# Lint
uv run python3 -m ruff check .
"
fi

if $FLAG_NODE; then
    CLAUDE_MD+="
# Install dependencies
npm install

# Run tests
npm test

# Lint
npm run lint
"
fi

CLAUDE_MD+="\`\`\`

---

## Architecture Summary

[Key components and their relationships]

| File/Directory | Purpose |
|---------------|---------|
| \`src/\` | Main source code |
| \`scripts/\` | Entry-point scripts |
| \`tests/\` | Test suite |

---

## Key Design Decisions

1. **[Decision 1]**: [Rationale]
2. **[Decision 2]**: [Rationale]

---

## Common Tasks

### Run Tests and Lint
\`\`\`bash"

if $FLAG_PYTHON || $FLAG_MINIMAL; then
    CLAUDE_MD+="
uv run python3 -m pytest tests/ -v
uv run python3 -m ruff check ."
fi

if $FLAG_NODE; then
    CLAUDE_MD+="
npm test
npm run lint"
fi

CLAUDE_MD+="
\`\`\`

---

## Testing & Linting

### Before Committing
\`\`\`bash"

if $FLAG_PYTHON || $FLAG_MINIMAL; then
    CLAUDE_MD+="
uv run python3 -m ruff check .
uv run python3 -m pytest tests/ -v"
fi

if $FLAG_NODE; then
    CLAUDE_MD+="
npm run lint
npm test"
fi

CLAUDE_MD+="
\`\`\`

### CI

GitHub Actions runs on every push/PR via \`.github/workflows/test.yml\`.

---

## File Structure

\`\`\`
${PKG_NAME}/
├── CLAUDE.md                 # This file"

if $FLAG_PYTHON; then
    CLAUDE_MD+="
├── pyproject.toml            # UV/hatchling project config
├── uv.lock                   # Lockfile (auto-generated)
├── src/
│   └── ${PKG_NAME}/
│       └── __init__.py"
fi

if $FLAG_NODE; then
    CLAUDE_MD+="
├── package.json
├── src/"
fi

CLAUDE_MD+="
├── scripts/                  # Entry-point scripts
├── tests/                    # Test suite
└── .github/workflows/        # CI configuration
\`\`\`

---

## Known Issues / TODOs

- [ ] [First TODO item]
- [ ] [Second TODO item]

---

## Resuming Work

When starting a new Claude session:

1. **Read this file**: Get project context
2. **Check git status**: \`git status && git log --oneline -5\`
3. **Run tests**: Verify nothing is broken"

if $FLAG_PYTHON || $FLAG_MINIMAL; then
    CLAUDE_MD+="
4. **Install deps**: \`uv sync --extra dev\`"
fi

if $FLAG_NODE; then
    CLAUDE_MD+="
4. **Install deps**: \`npm install\`"
fi

CLAUDE_MD+="
"

write_if_missing "CLAUDE.md" "$CLAUDE_MD"

# ---------------------------------------------------------------------------
# 4. .claude/settings.local.json
# ---------------------------------------------------------------------------

SETTINGS_ALLOW="[
      \"WebSearch\","

if $FLAG_PYTHON; then
    SETTINGS_ALLOW+="
      \"Bash(python3:*)\",
      \"Bash(uv:*)\","
fi

if $FLAG_NODE; then
    SETTINGS_ALLOW+="
      \"Bash(node:*)\",
      \"Bash(npm:*)\",
      \"Bash(npx:*)\","
fi

SETTINGS_ALLOW+="
      \"Bash(git clone:*)\",
      \"Bash(git checkout:*)\",
      \"Bash(git worktree:*)\",
      \"Bash(ls:*)\",
      \"Bash(find:*)\",
      \"Bash(wc:*)\",
      \"Bash(chmod:*)\"
    ]"

SETTINGS_CONTENT="{
  \"permissions\": {
    \"allow\": ${SETTINGS_ALLOW}
  }
}"

write_if_missing ".claude/settings.local.json" "$SETTINGS_CONTENT"

# ---------------------------------------------------------------------------
# 5. Claude Code memory directory
# ---------------------------------------------------------------------------

# Encode the current project path: /Users/foo/bar -> -Users-foo-bar
ENCODED_PATH=$(printf '%s' "$PROJECT_DIR" | tr '/' '-')

CLAUDE_HOME="${HOME}/.claude"
MEMORY_DIR="${CLAUDE_HOME}/projects/${ENCODED_PATH}/memory"

MEMORY_MD="# ${PROJECT_NAME} - Project Memory

## Key Facts
- Project initialized on $(date +%Y-%m-%d)
- [Add key facts as they emerge during development]

## Open Design Questions
- [Track unresolved architecture or implementation questions here]

## Deferred Items
- [Items intentionally postponed, with rationale]

## Session Notes
- [Record important decisions, gotchas, and context that should persist across sessions]
"

if [ ! -d "$MEMORY_DIR" ]; then
    mkdir -p "$MEMORY_DIR"
    printf "${GREEN}  + %s/${NC}\n" "$MEMORY_DIR"
fi

write_if_missing "${MEMORY_DIR}/MEMORY.md" "$MEMORY_MD"

# ---------------------------------------------------------------------------
# 6. Python project files (--python)
# ---------------------------------------------------------------------------

if $FLAG_PYTHON; then
    printf "\n${CYAN}Setting up Python project...${NC}\n"

    PYPROJECT="[project]
name = \"${PKG_NAME}\"
version = \"0.1.0\"
description = \"${PROJECT_NAME}\"
requires-python = \">=3.11\"
dependencies = []

[project.optional-dependencies]
dev = [
    \"ruff>=0.4.0\",
    \"pytest>=8.0.0\",
]

[build-system]
requires = [\"hatchling\"]
build-backend = \"hatchling.build\"

[tool.hatch.build.targets.wheel]
packages = [\"src\"]

[tool.ruff]
exclude = []

[tool.ruff.lint]
ignore = [\"E501\"]

[tool.pytest.ini_options]
testpaths = [\"tests\"]"

    write_if_missing "pyproject.toml" "$PYPROJECT"

    # src/<pkg>/__init__.py
    write_if_missing "src/${PKG_NAME}/__init__.py" "\"\"\"${PROJECT_NAME}.\"\"\"
"

    # tests/
    write_if_missing "tests/__init__.py" ""

    CONFTEST="\"\"\"Shared test fixtures.\"\"\"

import pytest  # noqa: F401
"
    write_if_missing "tests/conftest.py" "$CONFTEST"

    TEST_EXAMPLE="\"\"\"Example test to verify the test setup works.\"\"\"


def test_import():
    \"\"\"Verify the package is importable.\"\"\"
    import ${PKG_NAME}  # noqa: F401
"
    write_if_missing "tests/test_smoke.py" "$TEST_EXAMPLE"

    # scripts/
    if [ ! -d "scripts" ]; then
        mkdir -p scripts
        printf "${GREEN}  + scripts/${NC}\n"
    fi

    # GitHub Actions CI
    CI_PYTHON="name: Tests
on: [push, pull_request]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v5

      - name: Install dependencies
        run: uv sync --extra dev

      - name: Lint
        run: uv run python3 -m ruff check .

      - name: Test
        run: uv run python3 -m pytest tests/ -v"

    write_if_missing ".github/workflows/test.yml" "$CI_PYTHON"
fi

# ---------------------------------------------------------------------------
# 7. Node.js project files (--node)
# ---------------------------------------------------------------------------

if $FLAG_NODE; then
    printf "\n${CYAN}Setting up Node.js project...${NC}\n"

    PACKAGE_JSON="{
  \"name\": \"${PKG_NAME}\",
  \"version\": \"0.1.0\",
  \"description\": \"${PROJECT_NAME}\",
  \"main\": \"src/index.js\",
  \"scripts\": {
    \"start\": \"node src/index.js\",
    \"test\": \"jest --verbose\",
    \"lint\": \"eslint src/ tests/\",
    \"lint:fix\": \"eslint src/ tests/ --fix\"
  },
  \"devDependencies\": {
    \"eslint\": \"^9.0.0\",
    \"jest\": \"^29.0.0\"
  }
}"

    write_if_missing "package.json" "$PACKAGE_JSON"

    # src/
    write_if_missing "src/index.js" "// ${PROJECT_NAME} entry point
"

    # tests/
    TEST_NODE="// Smoke test to verify test setup
describe('Setup', () => {
  test('test runner works', () => {
    expect(1 + 1).toBe(2);
  });
});"

    write_if_missing "tests/smoke.test.js" "$TEST_NODE"

    # scripts/
    if [ ! -d "scripts" ]; then
        mkdir -p scripts
        printf "${GREEN}  + scripts/${NC}\n"
    fi

    # GitHub Actions CI for Node
    CI_NODE_FILE=".github/workflows/test.yml"
    if ! $FLAG_PYTHON; then
        # Only write Node CI if Python CI wasn't already written
        CI_NODE="name: Tests
on: [push, pull_request]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Lint
        run: npm run lint

      - name: Test
        run: npm test"

        write_if_missing "$CI_NODE_FILE" "$CI_NODE"
    fi
fi

# ---------------------------------------------------------------------------
# 8. Minimal mode creates directories only
# ---------------------------------------------------------------------------

if $FLAG_MINIMAL; then
    printf "\n${CYAN}Minimal mode: creating directory structure...${NC}\n"
    for dir in src scripts tests; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            printf "${GREEN}  + %s/${NC}\n" "$dir"
        fi
    done
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

printf "\n${CYAN}============================================${NC}\n"
printf "${CYAN}  Project initialized: %s${NC}\n" "$PROJECT_NAME"
printf "${CYAN}============================================${NC}\n\n"

if [ ${#created_files[@]} -eq 0 ]; then
    printf "No new files created (all already exist).\n"
else
    printf "Created %d files:\n" "${#created_files[@]}"
    for f in "${created_files[@]}"; do
        printf "  %s\n" "$f"
    done
fi

printf "\n${CYAN}Next steps:${NC}\n"

if $FLAG_PYTHON; then
    printf "  1. Install dependencies:  uv sync --extra dev\n"
    printf "  2. Run smoke test:        uv run python3 -m pytest tests/ -v\n"
    printf "  3. Run lint:              uv run python3 -m ruff check .\n"
fi

if $FLAG_NODE; then
    printf "  1. Install dependencies:  npm install\n"
    printf "  2. Run smoke test:        npm test\n"
    printf "  3. Run lint:              npm run lint\n"
fi

printf "  4. Edit CLAUDE.md with project-specific details\n"
printf "  5. Make your first commit!\n\n"

printf "${CYAN}Memory directory:${NC}\n"
printf "  %s/MEMORY.md\n\n" "$MEMORY_DIR"
