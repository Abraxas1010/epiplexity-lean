# Epiplexity Lean Module Map

## Module Dependency Graph

```
HeytingLean.Epiplexity
├── Prelude          (basic types: BitStr, FinDist re-exports)
├── Info             (Shannon entropy, NLL, cross-entropy)
│   └── depends on: Prelude
├── Programs         (Prog α, Feasible, codeLen)
│   └── depends on: Prelude, Info
├── MDL              (mdlCost, MDLinf)
│   └── depends on: Programs, Info
├── Core             (S_T, H_T, MDL_T definitions)
│   └── depends on: MDL, Programs
├── Bounds           (entropy bounds, Lemmas 15-16)
│   └── depends on: Core, Info
├── Conditional      (CondProg, conditional epiplexity)
│   └── depends on: Core, Programs
├── Emergence        (EpiplexityEmergent, STGap)
│   └── depends on: Conditional, Core
└── Crypto/
    ├── Axioms       (OWP, PRF, CSPRNG hypotheses)
    │   └── depends on: Prelude
    ├── HeavySet     (Heavy-set lemmas 6-8)
    │   └── depends on: Axioms, Core
    ├── CSPRNG       (Theorems 9, 12, 17-19)
    │   └── depends on: HeavySet, Core, Axioms
    ├── PRFHighEpiplexity (Theorem 24)
    │   └── depends on: HeavySet, Axioms
    └── Factorization (Theorems 13, 25, Corollary 26)
        └── depends on: HeavySet, Axioms, CSPRNG
```

## File Sizes

| File | Lines | Description |
|------|-------|-------------|
| Prelude.lean | ~50 | Basic type definitions |
| Info.lean | ~150 | Information theory |
| Programs.lean | ~100 | Program model |
| MDL.lean | ~120 | MDL cost functions |
| Core.lean | ~180 | Time-bounded measures |
| Bounds.lean | ~100 | Entropy bounds |
| Conditional.lean | ~200 | Conditional epiplexity |
| Emergence.lean | ~80 | Emergence predicate |
| Crypto/Axioms.lean | ~150 | Cryptographic axioms |
| Crypto/HeavySet.lean | ~200 | Heavy-set lemmas |
| Crypto/CSPRNG.lean | ~300 | CSPRNG theorems |
| Crypto/PRFHighEpiplexity.lean | ~180 | PRF theorem |
| Crypto/Factorization.lean | ~1200 | Factorization theorems |
| **Total** | **~3000** | |

## Key Imports from Mathlib

- `Mathlib.Probability.ProbabilityMassFunction` — Finite distributions
- `Mathlib.Analysis.SpecialFunctions.Log.Basic` — Logarithms for entropy
- `Mathlib.Data.Real.Basic` — Real numbers
- `Mathlib.Data.Fintype.Basic` — Finite types
- `Mathlib.Order.ConditionallyCompleteLattice` — Infimum for MDL
