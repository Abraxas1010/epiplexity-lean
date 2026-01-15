# Reproducibility Guide

## Prerequisites

- **Lean 4**: Version 4.24.0 (see `lean-toolchain`)
- **Lake**: Bundled with Lean
- **elan**: Recommended for toolchain management

## Quick Start

```bash
# Clone or navigate to the repository
cd RESEARCHER_BUNDLE

# Build all modules
lake build

# Verify no sorry/admit (from monorepo root)
grep -r "sorry\|admit" HeytingLean/Epiplexity/ --include="*.lean"
# Should return empty
```

## Full Verification

```bash
# From RESEARCHER_BUNDLE directory
lake build --wfail

# Expected output: Build successful with 0 warnings
```

## Regenerating Visualizations

```bash
# Generate UMAP preview SVGs
node scripts/render_umap_previews.js

# Output:
# - artifacts/visuals/epiplexity_2d_preview.svg
# - artifacts/visuals/epiplexity_3d_preview.svg
# - artifacts/visuals/epiplexity_3d_preview_animated.svg
```

## Troubleshooting

### Mathlib Cache

If the build is slow, try fetching the Mathlib cache:

```bash
lake exe cache get
```

### Toolchain Mismatch

Ensure your toolchain matches:

```bash
cat lean-toolchain
# Should show: leanprover/lean4:v4.24.0

elan show
# Should show the matching version
```

### Clean Build

If incremental build fails:

```bash
lake clean
lake build
```

## Build Artifacts

After successful build:

```
RESEARCHER_BUNDLE/
├── .lake/
│   ├── build/lib/HeytingLean/Epiplexity/  # Compiled .olean files
│   └── packages/                           # Mathlib and dependencies
├── artifacts/
│   └── visuals/
│       ├── epiplexity_2d_preview.svg
│       ├── epiplexity_3d_preview.svg
│       └── epiplexity_3d_preview_animated.svg
└── reports/
    └── build_log.txt                       # Build output (if captured)
```
