# Dependencies

## Lean Toolchain

```
leanprover/lean4:v4.24.0
```

## Mathlib

Pinned to commit: `f897ebcf72cd16f89ab4577d0c826cd14afaafc7`

## Transitive Dependencies

| Package | Version/Commit | Purpose |
|---------|----------------|---------|
| mathlib4 | f897ebcf... | Core mathematics library |
| batteries | 8da40b72... | Extended stdlib |
| aesop | 725ac8cd... | Automated reasoning |
| Qq | dea6a336... | Quote/antiquote |
| proofwidgets | 556caed0... | Proof visualization |
| plausible | dfd06ebf... | Property testing |
| LeanSearchClient | 99657ad9... | Search integration |
| importGraph | d7681268... | Import visualization |

## Key Mathlib Imports

### Probability Theory
- `Mathlib.Probability.ProbabilityMassFunction.Basic`
- `Mathlib.Probability.ProbabilityMassFunction.Constructions`

### Analysis
- `Mathlib.Analysis.SpecialFunctions.Log.Basic`
- `Mathlib.Analysis.SpecialFunctions.Pow.Real`

### Algebra
- `Mathlib.Algebra.Order.Field.Basic`
- `Mathlib.Algebra.BigOperators.Finsupp`

### Data Types
- `Mathlib.Data.Real.Basic`
- `Mathlib.Data.Fintype.Basic`
- `Mathlib.Data.Finset.Basic`

### Order Theory
- `Mathlib.Order.ConditionallyCompleteLattice.Basic`
- `Mathlib.Order.Bounds.Basic`

## License Compatibility

All dependencies are Apache 2.0 or compatible licenses.
