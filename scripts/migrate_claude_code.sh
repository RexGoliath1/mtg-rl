#!/usr/bin/env bash
# =============================================================================
# migrate_claude_code.sh — Transfer Claude Code environment to a new server
# =============================================================================
#
# Usage:
#   ./scripts/migrate_claude_code.sh --export                 # On source machine
#   ./scripts/migrate_claude_code.sh --import <tarball> <repo> # On destination
#   ./scripts/migrate_claude_code.sh --verify <repo>           # Post-import check
#
# Examples:
#   # 1. Export on your Mac
#   ./scripts/migrate_claude_code.sh --export
#
#   # 2. Copy tarball to new server
#   scp claude_code_export.tar.gz ubuntu@server:~/
#
#   # 3. Import on the new server (repo already cloned)
#   ./scripts/migrate_claude_code.sh --import ~/claude_code_export.tar.gz ~/git/mtg
#
#   # 4. Verify everything works
#   ./scripts/migrate_claude_code.sh --verify ~/git/mtg
#
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers (degrade gracefully when stdout is not a terminal)
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' BOLD='' NC=''
fi

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }
header(){ echo -e "\n${BOLD}=== $* ===${NC}"; }

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    cat <<'USAGE'
Usage:
  migrate_claude_code.sh --export
  migrate_claude_code.sh --import <tarball> <repo_path>
  migrate_claude_code.sh --verify <repo_path>

Modes:
  --export   Package Claude Code config + memory into a tarball (run on source)
  --import   Unpack tarball into correct locations on a new server
  --verify   Check that all files and packages are in place after import

USAGE
    exit 1
}

# ---------------------------------------------------------------------------
# MODE 1: EXPORT (run on source machine)
# ---------------------------------------------------------------------------
do_export() {
    header "Claude Code Environment Export"

    local repo_root
    repo_root="$(cd "$(dirname "$0")/.." && pwd)"
    local claude_home="$HOME/.claude"

    # Derive the project-specific memory directory from the repo path.
    # Claude Code encodes the absolute path by replacing "/" with "-"
    # and stripping the leading "-".
    local encoded_path
    encoded_path="$(echo "$repo_root" | tr '/' '-' | sed 's/^-//')"
    local memory_dir="$claude_home/projects/-${encoded_path}/memory"

    local staging
    staging="$(mktemp -d)"
    trap 'rm -rf "$staging"' EXIT

    mkdir -p "$staging/global" "$staging/memory" "$staging/project"

    local missing=0

    # ---- Global config ----
    info "Collecting global Claude config..."

    if [ -f "$claude_home/CLAUDE.md" ]; then
        cp "$claude_home/CLAUDE.md" "$staging/global/CLAUDE.md"
        ok "global/CLAUDE.md ($(wc -c < "$claude_home/CLAUDE.md" | tr -d ' ') bytes)"
    else
        warn "~/.claude/CLAUDE.md not found — skipping"
        missing=$((missing + 1))
    fi

    if [ -f "$claude_home/settings.json" ]; then
        cp "$claude_home/settings.json" "$staging/global/settings.json"
        ok "global/settings.json"
    else
        warn "~/.claude/settings.json not found — skipping"
        missing=$((missing + 1))
    fi

    # ---- Project memory files ----
    info "Collecting project memory files from $memory_dir ..."

    for memfile in MEMORY.md user.md archived_docs.md; do
        if [ -f "$memory_dir/$memfile" ]; then
            cp "$memory_dir/$memfile" "$staging/memory/$memfile"
            ok "memory/$memfile ($(wc -c < "$memory_dir/$memfile" | tr -d ' ') bytes)"
        else
            warn "memory/$memfile not found — skipping"
            missing=$((missing + 1))
        fi
    done

    # ---- Project-level files ----
    info "Collecting project-level files..."

    if [ -f "$repo_root/.env" ]; then
        cp "$repo_root/.env" "$staging/project/env"
        ok "project/env (CONTAINS SECRETS — handle with care)"
    else
        warn ".env not found in repo root — skipping"
        missing=$((missing + 1))
    fi

    if [ -f "$repo_root/.claude/settings.local.json" ]; then
        cp "$repo_root/.claude/settings.local.json" "$staging/project/settings.local.json"
        ok "project/settings.local.json"
    else
        warn ".claude/settings.local.json not found — skipping"
        missing=$((missing + 1))
    fi

    # ---- Create tarball ----
    local tarball="$repo_root/claude_code_export.tar.gz"
    tar -czf "$tarball" -C "$staging" .

    header "Export Complete"
    echo ""
    echo "Tarball: $tarball"
    echo "Size:    $(wc -c < "$tarball" | tr -d ' ') bytes"
    if [ "$missing" -gt 0 ]; then
        warn "$missing file(s) were missing and skipped"
    fi
    echo ""

    # ---- Contents listing ----
    info "Contents:"
    tar -tzf "$tarball" | sort | while read -r f; do
        echo "  $f"
    done

    echo ""
    header "Next Steps"
    echo "  1. Copy tarball to the new server:"
    echo "     scp $tarball user@server:~/"
    echo ""
    echo "  2. Clone the repo on the new server:"
    echo "     git clone git@github.com:<org>/mtg.git ~/git/mtg"
    echo ""
    echo "  3. Run import on the new server:"
    echo "     ./scripts/migrate_claude_code.sh --import ~/claude_code_export.tar.gz ~/git/mtg"
    echo ""
    echo "  4. Verify:"
    echo "     ./scripts/migrate_claude_code.sh --verify ~/git/mtg"
    echo ""
    warn "The tarball contains .env secrets. Delete it after import."
}

# ---------------------------------------------------------------------------
# MODE 2: IMPORT (run on destination machine)
# ---------------------------------------------------------------------------
do_import() {
    local tarball="${1:-}"
    local repo_path="${2:-}"

    if [ -z "$tarball" ] || [ -z "$repo_path" ]; then
        fail "Usage: migrate_claude_code.sh --import <tarball> <repo_path>"
        exit 1
    fi

    if [ ! -f "$tarball" ]; then
        fail "Tarball not found: $tarball"
        exit 1
    fi

    # Resolve to absolute path
    repo_path="$(cd "$repo_path" 2>/dev/null && pwd)" || {
        fail "Repo path does not exist: $repo_path"
        exit 1
    }

    header "Claude Code Environment Import"
    info "Tarball:   $tarball"
    info "Repo path: $repo_path"

    local claude_home="$HOME/.claude"

    # Compute the project-encoded path for the DESTINATION repo location.
    local encoded_path
    encoded_path="$(echo "$repo_path" | tr '/' '-' | sed 's/^-//')"
    local project_dir="$claude_home/projects/-${encoded_path}"
    local memory_dir="$project_dir/memory"

    info "Project memory dir: $memory_dir"

    # ---- Extract tarball to temp dir ----
    local staging
    staging="$(mktemp -d)"
    trap 'rm -rf "$staging"' EXIT
    tar -xzf "$tarball" -C "$staging"

    # ---- Install global config ----
    header "Installing Global Config"
    mkdir -p "$claude_home"

    if [ -f "$staging/global/CLAUDE.md" ]; then
        if [ -f "$claude_home/CLAUDE.md" ]; then
            warn "~/.claude/CLAUDE.md already exists — backing up to CLAUDE.md.bak"
            cp "$claude_home/CLAUDE.md" "$claude_home/CLAUDE.md.bak"
        fi
        cp "$staging/global/CLAUDE.md" "$claude_home/CLAUDE.md"
        ok "Installed ~/.claude/CLAUDE.md"
    else
        warn "global/CLAUDE.md not in tarball — skipping"
    fi

    if [ -f "$staging/global/settings.json" ]; then
        if [ -f "$claude_home/settings.json" ]; then
            warn "~/.claude/settings.json already exists — backing up to settings.json.bak"
            cp "$claude_home/settings.json" "$claude_home/settings.json.bak"
        fi
        cp "$staging/global/settings.json" "$claude_home/settings.json"
        ok "Installed ~/.claude/settings.json"
    else
        warn "global/settings.json not in tarball — skipping"
    fi

    # ---- Install memory files ----
    header "Installing Project Memory"
    mkdir -p "$memory_dir"

    for memfile in MEMORY.md user.md archived_docs.md; do
        if [ -f "$staging/memory/$memfile" ]; then
            cp "$staging/memory/$memfile" "$memory_dir/$memfile"
            ok "Installed $memory_dir/$memfile"
        else
            warn "$memfile not in tarball — skipping"
        fi
    done

    # ---- Install project files ----
    header "Installing Project Files"

    if [ -f "$staging/project/env" ]; then
        if [ -f "$repo_path/.env" ]; then
            warn ".env already exists — backing up to .env.bak"
            cp "$repo_path/.env" "$repo_path/.env.bak"
        fi
        cp "$staging/project/env" "$repo_path/.env"
        chmod 600 "$repo_path/.env"
        ok "Installed $repo_path/.env (mode 600)"
    else
        warn "project/env not in tarball — .env skipped"
    fi

    if [ -f "$staging/project/settings.local.json" ]; then
        mkdir -p "$repo_path/.claude"
        cp "$staging/project/settings.local.json" "$repo_path/.claude/settings.local.json"
        ok "Installed $repo_path/.claude/settings.local.json"
    else
        warn "project/settings.local.json not in tarball — skipping"
    fi

    # ---- Install Python dependencies ----
    header "Installing Python Dependencies"
    if command -v uv &>/dev/null; then
        info "Running uv sync --all-extras in $repo_path ..."
        (cd "$repo_path" && uv sync --all-extras)
        ok "Python dependencies installed"
    else
        warn "uv not found — install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
        warn "Then run: cd $repo_path && uv sync --all-extras"
    fi

    # ---- Regenerate data files ----
    header "Regenerating Data Files"
    info "This may take a few minutes (Scryfall download + embedding computation)."

    mkdir -p "$repo_path/data"

    # Step 1: Download Scryfall bulk data
    info "Step 1/4: Downloading Scryfall bulk data..."
    local bulk_json="$repo_path/data/scryfall_bulk_cards.json"
    if [ -f "$bulk_json" ]; then
        info "Scryfall bulk JSON already exists — skipping download"
    else
        local bulk_url
        bulk_url=$(curl -s https://api.scryfall.com/bulk-data/default-cards \
            | python3 -c "import sys,json; print(json.load(sys.stdin)['download_uri'])" 2>/dev/null) || true

        if [ -n "${bulk_url:-}" ]; then
            info "Downloading from $bulk_url ..."
            curl -L --progress-bar -o "$bulk_json" "$bulk_url"
            ok "Downloaded Scryfall bulk data ($(du -h "$bulk_json" | cut -f1))"
        else
            warn "Could not fetch Scryfall download URL — run manually later:"
            warn "  curl -s https://api.scryfall.com/bulk-data/default-cards | python3 -c \"import sys,json; print(json.load(sys.stdin)['download_uri'])\""
        fi
    fi

    # Step 2: Precompute HDF5 embeddings
    info "Step 2/4: Pre-computing card mechanics embeddings..."
    if [ -f "$bulk_json" ]; then
        (cd "$repo_path" && uv run python3 -m src.mechanics.precompute_embeddings \
            --format commander --bulk-json "$bulk_json") && \
            ok "HDF5 embeddings generated" || \
            warn "Embedding precompute failed — run manually later"
    else
        warn "Scryfall bulk JSON missing — skipping embedding precompute"
    fi

    # Step 3: Generate recommender sidecar
    info "Step 3/4: Generating recommender sidecar metadata..."
    if [ -f "$bulk_json" ]; then
        (cd "$repo_path" && uv run python3 scripts/generate_recommender_sidecar.py) && \
            ok "Recommender sidecar generated" || \
            warn "Sidecar generation failed — run manually later"
    else
        warn "Scryfall bulk JSON missing — skipping sidecar generation"
    fi

    # Step 4: Generate sample collection
    info "Step 4/4: Generating sample collection..."
    (cd "$repo_path" && uv run python3 scripts/generate_sample_collection.py) && \
        ok "Sample collection generated" || \
        warn "Sample collection generation failed — run manually later"

    # ---- Summary ----
    header "Import Complete"
    echo ""
    echo "Files installed:"
    echo "  ~/.claude/CLAUDE.md"
    echo "  ~/.claude/settings.json"
    echo "  $memory_dir/MEMORY.md"
    echo "  $memory_dir/user.md"
    echo "  $memory_dir/archived_docs.md"
    echo "  $repo_path/.env"
    echo "  $repo_path/.claude/settings.local.json"
    echo ""
    echo "Data regenerated:"
    echo "  $repo_path/data/scryfall_bulk_cards.json"
    echo "  $repo_path/data/card_mechanics_commander.h5"
    echo "  $repo_path/data/card_recommender_metadata.json"
    echo "  $repo_path/data/sample_collection.csv"
    echo ""

    header "Post-Import Checklist"
    echo "  1. Run verify:  ./scripts/migrate_claude_code.sh --verify $repo_path"
    echo "  2. Run tests:   cd $repo_path && uv run python3 -m pytest tests/test_parser.py -v"
    echo "  3. Run lint:    cd $repo_path && uv run python3 -m ruff check . --ignore E501 --exclude forge-repo,wandb"
    echo "  4. Update .env with new server credentials if needed"
    echo "  5. Delete the tarball: rm $tarball"
    echo ""
    warn "Remember: the tarball contains .env secrets. Delete it now if possible."
}

# ---------------------------------------------------------------------------
# MODE 3: VERIFY (run on destination after import)
# ---------------------------------------------------------------------------
do_verify() {
    local repo_path="${1:-}"

    if [ -z "$repo_path" ]; then
        fail "Usage: migrate_claude_code.sh --verify <repo_path>"
        exit 1
    fi

    repo_path="$(cd "$repo_path" 2>/dev/null && pwd)" || {
        fail "Repo path does not exist: $repo_path"
        exit 1
    }

    header "Claude Code Environment Verification"
    info "Repo: $repo_path"

    local claude_home="$HOME/.claude"
    local encoded_path
    encoded_path="$(echo "$repo_path" | tr '/' '-' | sed 's/^-//')"
    local memory_dir="$claude_home/projects/-${encoded_path}/memory"

    local pass=0
    local total=0

    check_file() {
        local label="$1"
        local filepath="$2"
        local check_content="${3:-false}"
        total=$((total + 1))

        if [ ! -f "$filepath" ]; then
            fail "$label: NOT FOUND ($filepath)"
            return
        fi

        local size
        size="$(wc -c < "$filepath" | tr -d ' ')"
        if [ "$size" -eq 0 ]; then
            fail "$label: EXISTS BUT EMPTY ($filepath)"
            return
        fi

        if [ "$check_content" = "readable" ]; then
            if head -1 "$filepath" &>/dev/null; then
                ok "$label: OK ($size bytes)"
                pass=$((pass + 1))
            else
                fail "$label: EXISTS BUT NOT READABLE ($filepath)"
            fi
        else
            ok "$label: OK ($size bytes)"
            pass=$((pass + 1))
        fi
    }

    # ---- Global config ----
    header "Global Config"
    check_file "~/.claude/CLAUDE.md" "$claude_home/CLAUDE.md" readable
    check_file "~/.claude/settings.json" "$claude_home/settings.json"

    # ---- Memory files ----
    header "Project Memory ($memory_dir)"
    check_file "MEMORY.md" "$memory_dir/MEMORY.md" readable
    check_file "user.md" "$memory_dir/user.md" readable
    check_file "archived_docs.md" "$memory_dir/archived_docs.md" readable

    # ---- Project files ----
    header "Project Files"
    check_file "CLAUDE.md (repo)" "$repo_path/CLAUDE.md" readable
    check_file ".env" "$repo_path/.env"
    check_file ".claude/settings.local.json" "$repo_path/.claude/settings.local.json"

    # ---- Data files ----
    header "Data Files"
    check_file "scryfall_bulk_cards.json" "$repo_path/data/scryfall_bulk_cards.json"
    check_file "card_mechanics_commander.h5" "$repo_path/data/card_mechanics_commander.h5"
    check_file "card_recommender_metadata.json" "$repo_path/data/card_recommender_metadata.json"
    check_file "card_color_identity.json" "$repo_path/data/card_color_identity.json"
    check_file "sample_collection.csv" "$repo_path/data/sample_collection.csv"

    # ---- Package import test ----
    header "Package Import Test"
    total=$((total + 1))
    if (cd "$repo_path" && uv run python3 -c "import src.mechanics.vocabulary; print(f'VOCAB_SIZE={src.mechanics.vocabulary.VOCAB_SIZE}')" 2>/dev/null); then
        ok "src.mechanics.vocabulary imports successfully"
        pass=$((pass + 1))
    else
        fail "Could not import src.mechanics.vocabulary"
    fi

    total=$((total + 1))
    if (cd "$repo_path" && uv run python3 -c "from src.mechanics.card_parser import encode_card; print('card_parser OK')" 2>/dev/null); then
        ok "src.mechanics.card_parser imports successfully"
        pass=$((pass + 1))
    else
        fail "Could not import src.mechanics.card_parser"
    fi

    # ---- Summary ----
    header "Verification Summary"
    if [ "$pass" -eq "$total" ]; then
        ok "All $total checks passed"
    else
        local failed=$((total - pass))
        warn "$pass/$total passed, $failed failed"
        echo ""
        echo "To fix missing data files, run:"
        echo "  cd $repo_path"
        echo "  uv run python3 -m src.mechanics.precompute_embeddings --format commander --bulk-json data/scryfall_bulk_cards.json"
        echo "  uv run python3 scripts/generate_recommender_sidecar.py"
        echo "  uv run python3 scripts/generate_sample_collection.py"
    fi
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
case "${1:-}" in
    --export)
        do_export
        ;;
    --import)
        shift
        do_import "$@"
        ;;
    --verify)
        shift
        do_verify "$@"
        ;;
    *)
        usage
        ;;
esac
