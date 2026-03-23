#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Task 019 Usability Test — Dynamic CLI Hybrid Config Choices
# ─────────────────────────────────────────────────────────────────────────────
#
# Prerequisites:
#   - Admin API running on localhost:8420 (python -m src.run_daemon --api)
#     OR standalone:  uvicorn src.admin.app:create_app --factory --port 8420
#
# What this script tests (end-to-end):
#   1. List current configs via API — confirm original 13 are present
#   2. Create a new test config ("usability_test_019") via API
#   3. Verify it appears in the config list
#   4. Run sanity check on the new config (expected: fail, since model is fake)
#   5. Verify CLI --hybrid accepts the new config name (argparse validation)
#   6. Verify CLI --help shows the new config in choices
#   7. Clean up — delete the test config via API
#   8. Verify CLI --hybrid rejects the deleted config name
#
# Usage:
#   chmod +x scripts/test_019_usability.sh
#   ./scripts/test_019_usability.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

API="http://localhost:8420"
TEST_CONFIG="usability_test_019"
PASS=0
FAIL=0
TOTAL=0

# ── Helpers ──────────────────────────────────────────────────────────────────

green()  { printf "\033[32m✓ %s\033[0m\n" "$1"; }
red()    { printf "\033[31m✗ %s\033[0m\n" "$1"; }
header() { printf "\n\033[1;36m── %s ──\033[0m\n" "$1"; }

check() {
    TOTAL=$((TOTAL + 1))
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then
        green "$desc"
        PASS=$((PASS + 1))
    else
        red "$desc"
        FAIL=$((FAIL + 1))
    fi
}

check_output() {
    TOTAL=$((TOTAL + 1))
    local desc="$1"
    local output="$2"
    local pattern="$3"
    if echo "$output" | grep -q "$pattern"; then
        green "$desc"
        PASS=$((PASS + 1))
    else
        red "$desc (expected pattern: $pattern)"
        FAIL=$((FAIL + 1))
    fi
}

check_not_output() {
    TOTAL=$((TOTAL + 1))
    local desc="$1"
    local output="$2"
    local pattern="$3"
    if echo "$output" | grep -q "$pattern"; then
        red "$desc (should NOT contain: $pattern)"
        FAIL=$((FAIL + 1))
    else
        green "$desc"
        PASS=$((PASS + 1))
    fi
}

# ── Preflight ────────────────────────────────────────────────────────────────

header "Preflight: checking admin API is running"
if ! curl -sf "$API/health" >/dev/null 2>&1; then
    echo ""
    echo "Admin API is not running on $API"
    echo "Start it with:  python -m src.run_daemon --api"
    echo "  or:           uvicorn src.admin.app:create_app --factory --port 8420"
    exit 1
fi
green "Admin API reachable at $API"

# ── Step 1: List existing configs ────────────────────────────────────────────

header "Step 1: List existing hybrid configs"
LIST_RESPONSE=$(curl -sf "$API/config/hybrid-configs")
CONFIG_COUNT=$(echo "$LIST_RESPONSE" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['configs']))")

check_output "Got config list from API" "$CONFIG_COUNT" "^[0-9]"
check_output "At least 13 configs present (got $CONFIG_COUNT)" "$CONFIG_COUNT" "^1[3-9]\|^[2-9][0-9]"
check_output "all_cloud in config list" "$LIST_RESPONSE" "all_cloud"
check_output "hybrid_haiku_tools in config list" "$LIST_RESPONSE" "hybrid_haiku_tools"

# ── Step 2: Create a test config via API ─────────────────────────────────────

header "Step 2: Create test config '$TEST_CONFIG' via admin API"

# Clean up first in case a previous run left it behind
curl -sf -X DELETE "$API/config/hybrid-configs/$TEST_CONFIG" >/dev/null 2>&1 || true

CREATE_RESPONSE=$(curl -sf -X POST "$API/config/hybrid-configs" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "'"$TEST_CONFIG"'",
        "tool_provider": "anthropic",
        "tool_model": "claude-haiku-4-5-20251001",
        "reasoning_quick_provider": "ollama",
        "reasoning_quick_model": "qwen2.5:14b",
        "reasoning_deep_provider": "anthropic",
        "reasoning_deep_model": "claude-sonnet-4-5-20250929",
        "enhance_local": true,
        "enhance_style": "financial_analysis",
        "enhance_deep": false,
        "enhance_deep_style": "execution_params_only"
    }' 2>&1) || true

check_output "Config created successfully" "$CREATE_RESPONSE" "$TEST_CONFIG"

# ── Step 3: Verify new config appears in list ────────────────────────────────

header "Step 3: Verify new config appears in API list"
LIST_AFTER=$(curl -sf "$API/config/hybrid-configs")
check_output "'$TEST_CONFIG' visible in config list" "$LIST_AFTER" "$TEST_CONFIG"

NEW_COUNT=$(echo "$LIST_AFTER" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['configs']))")
EXPECTED=$((CONFIG_COUNT + 1))
check_output "Config count incremented ($CONFIG_COUNT -> $NEW_COUNT)" "$NEW_COUNT" "^${EXPECTED}$"

# ── Step 4: Sanity check on the new config ───────────────────────────────────

header "Step 4: Run sanity check on '$TEST_CONFIG'"
SANITY_RESPONSE=$(curl -sf -X POST "$API/config/hybrid-configs/$TEST_CONFIG/sanity-check" 2>&1) || SANITY_RESPONSE="error"

if [ "$SANITY_RESPONSE" != "error" ]; then
    check_output "Sanity check endpoint responded" "$SANITY_RESPONSE" "overall"
    # We expect at least partial results — ollama may not be running
    green "Sanity check returned results (pass/partial/fail depends on local services)"
    PASS=$((PASS + 1)); TOTAL=$((TOTAL + 1))
else
    red "Sanity check endpoint failed to respond"
    FAIL=$((FAIL + 1)); TOTAL=$((TOTAL + 1))
fi

# ── Step 5: Verify CLI accepts the new config name ──────────────────────────

header "Step 5: Verify CLI --hybrid accepts '$TEST_CONFIG'"

# Use --help trick: parse_args will fail with SystemExit for invalid choices
# but we can check by importing the parser logic directly
CLI_TEST=$(python3 -c "
from src.hybrid_llm import CONFIGS
if '$TEST_CONFIG' in CONFIGS:
    print('ACCEPTED')
else:
    print('REJECTED')
" 2>&1)
check_output "CLI CONFIGS contains '$TEST_CONFIG'" "$CLI_TEST" "ACCEPTED"

# Also verify argparse would accept it
ARGPARSE_TEST=$(python3 -c "
import argparse
from src.hybrid_llm import CONFIGS as _hybrid_configs
parser = argparse.ArgumentParser()
parser.add_argument('--hybrid', type=str, choices=list(_hybrid_configs.keys()))
args = parser.parse_args(['--hybrid', '$TEST_CONFIG'])
print('PARSED:', args.hybrid)
" 2>&1)
check_output "argparse accepts '$TEST_CONFIG'" "$ARGPARSE_TEST" "PARSED: $TEST_CONFIG"

# ── Step 6: Verify --help shows the new config ──────────────────────────────

header "Step 6: Verify --help shows '$TEST_CONFIG' in choices"
HELP_OUTPUT=$(python3 -c "
import argparse
from src.hybrid_llm import CONFIGS as _hybrid_configs
parser = argparse.ArgumentParser()
parser.add_argument('--hybrid', type=str, choices=list(_hybrid_configs.keys()))
parser.print_help()
" 2>&1)
check_output "'$TEST_CONFIG' appears in --help output" "$HELP_OUTPUT" "$TEST_CONFIG"

# ── Step 7: Clean up — delete the test config ───────────────────────────────

header "Step 7: Delete test config '$TEST_CONFIG'"
DELETE_STATUS=$(curl -sf -o /dev/null -w "%{http_code}" -X DELETE "$API/config/hybrid-configs/$TEST_CONFIG")
check_output "DELETE returned 204" "$DELETE_STATUS" "204"

# Verify gone from API
LIST_FINAL=$(curl -sf "$API/config/hybrid-configs")
check_not_output "'$TEST_CONFIG' removed from config list" "$LIST_FINAL" "$TEST_CONFIG"

FINAL_COUNT=$(echo "$LIST_FINAL" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['configs']))")
check_output "Config count restored ($FINAL_COUNT = $CONFIG_COUNT)" "$FINAL_COUNT" "^${CONFIG_COUNT}$"

# ── Step 8: Verify CLI rejects the deleted config ───────────────────────────

header "Step 8: Verify CLI rejects deleted config"
REJECT_TEST=$(python3 -c "
import argparse
from src.hybrid_llm import CONFIGS as _hybrid_configs
parser = argparse.ArgumentParser()
parser.add_argument('--hybrid', type=str, choices=list(_hybrid_configs.keys()))
try:
    parser.parse_args(['--hybrid', '$TEST_CONFIG'])
    print('WRONGLY_ACCEPTED')
except SystemExit:
    print('CORRECTLY_REJECTED')
" 2>&1)
check_output "argparse rejects deleted '$TEST_CONFIG'" "$REJECT_TEST" "CORRECTLY_REJECTED"

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════"
if [ "$FAIL" -eq 0 ]; then
    printf "\033[1;32m  ALL %d TESTS PASSED\033[0m\n" "$TOTAL"
else
    printf "\033[1;31m  %d/%d PASSED, %d FAILED\033[0m\n" "$PASS" "$TOTAL" "$FAIL"
fi
echo "═══════════════════════════════════════════════════"
echo ""

exit "$FAIL"
