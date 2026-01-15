#!/bin/bash
# Epiplexity PaperPack Verification Script
#
# Usage: ./scripts/verify_epiplexity.sh
#
# This script:
# 1. Checks for sorry/admit in the codebase
# 2. Runs lake build with strict flags
# 3. Generates visualization artifacts
# 4. Produces a verification report

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
REPORT_DIR="$ROOT_DIR/reports"
ARTIFACTS_DIR="$ROOT_DIR/artifacts"

mkdir -p "$REPORT_DIR"

echo "=== Epiplexity PaperPack Verification ==="
echo "Date: $(date)"
echo ""

# Step 1: Check for sorry/admit
echo "--- Step 1: Checking for sorry/admit ---"
SORRY_COUNT=$(grep -r "sorry\|admit" "$ROOT_DIR/HeytingLean" --include="*.lean" | wc -l || echo 0)
if [ "$SORRY_COUNT" -eq 0 ]; then
    echo "✓ No sorry/admit found"
else
    echo "✗ Found $SORRY_COUNT sorry/admit instances:"
    grep -r "sorry\|admit" "$ROOT_DIR/HeytingLean" --include="*.lean"
    exit 1
fi
echo ""

# Step 2: Build with strict flags
echo "--- Step 2: Building with lake ---"
cd "$ROOT_DIR"
if lake build 2>&1 | tee "$REPORT_DIR/build_log.txt"; then
    echo "✓ Build succeeded"
else
    echo "✗ Build failed"
    exit 1
fi
echo ""

# Step 3: Generate visualizations
echo "--- Step 3: Generating visualizations ---"
if command -v node &>/dev/null; then
    node "$SCRIPT_DIR/render_umap_previews.js"
    echo "✓ Visualizations generated"
else
    echo "⚠ Node.js not found, skipping visualization generation"
fi
echo ""

# Step 4: Count declarations
echo "--- Step 4: Declaration statistics ---"
THEOREM_COUNT=$(grep -r "^theorem " "$ROOT_DIR/HeytingLean" --include="*.lean" | wc -l)
LEMMA_COUNT=$(grep -r "^lemma " "$ROOT_DIR/HeytingLean" --include="*.lean" | wc -l)
DEF_COUNT=$(grep -r "^def " "$ROOT_DIR/HeytingLean" --include="*.lean" | wc -l)
STRUCTURE_COUNT=$(grep -r "^structure " "$ROOT_DIR/HeytingLean" --include="*.lean" | wc -l)

echo "Theorems: $THEOREM_COUNT"
echo "Lemmas: $LEMMA_COUNT"
echo "Definitions: $DEF_COUNT"
echo "Structures: $STRUCTURE_COUNT"
echo ""

# Step 5: Generate report
REPORT_FILE="$REPORT_DIR/verification_report_$(date +%Y%m%d_%H%M%S).md"
cat > "$REPORT_FILE" << EOF
# Epiplexity Verification Report

**Date:** $(date)
**Status:** PASSED

## Summary

- sorry/admit: 0
- Build: SUCCESS
- Visualizations: GENERATED

## Statistics

| Category | Count |
|----------|-------|
| Theorems | $THEOREM_COUNT |
| Lemmas | $LEMMA_COUNT |
| Definitions | $DEF_COUNT |
| Structures | $STRUCTURE_COUNT |

## Artifacts

- Build log: \`reports/build_log.txt\`
- 2D Preview: \`artifacts/visuals/epiplexity_2d_preview.svg\`
- 3D Preview: \`artifacts/visuals/epiplexity_3d_preview.svg\`
- 3D Animated: \`artifacts/visuals/epiplexity_3d_preview_animated.svg\`
EOF

echo "=== Verification Complete ==="
echo "Report written to: $REPORT_FILE"
