# Epiplexity Proof Index

## Main Theorems (from Finzi et al., 2026)

| Paper Ref | Lean Name | Statement | Module |
|-----------|-----------|-----------|--------|
| **Theorem 9** | `theorem9_CSPRNG_high_epiplexity` | CSPRNG → High Epiplexity | Crypto/CSPRNG |
| **Theorem 12** | `theorem12_CSPRNGBeta_conditional` | CSPRNGβ → β-conditional high epiplexity | Crypto/CSPRNG |
| **Theorem 13** | `theorem13` | OWP → Factorization hardness (weaker form) | Crypto/Factorization |
| **Theorem 17** | `theorem17_CSPRNG_poly_gap` | CSPRNG ↔ polynomial STGap | Crypto/CSPRNG |
| **Theorem 18** | `theorem18_CSPRNG_superpoly_gap` | CSPRNG ↔ super-polynomial STGap | Crypto/CSPRNG |
| **Theorem 19** | `theorem19_CSPRNG_characterization` | CSPRNG complete characterization | Crypto/CSPRNG |
| **Theorem 24** | `theorem24_PRF_high_epiplexity` | PRF → High Epiplexity | Crypto/PRFHighEpiplexity |
| **Theorem 25** | `theorem25` | OWP → Factorization hardness (main) | Crypto/Factorization |
| **Corollary 26** | `corollary26` | OWP → Average-case factorization | Crypto/Factorization |

## Supporting Lemmas

| Paper Ref | Lean Name | Statement | Module |
|-----------|-----------|-----------|--------|
| **Lemma 6** | `lemma6_heavy_set_exists` | Heavy-set existence | Crypto/HeavySet |
| **Lemma 7** | `lemma7_heavy_set_probability` | Heavy-set probability bound | Crypto/HeavySet |
| **Lemma 8** | `lemma8_heavy_set_compression` | Heavy-set compression | Crypto/HeavySet |
| **Lemma 15** | `lemma15_MDLinf_le` | MDL∞ upper bound | Bounds |
| **Lemma 16** | `lemma16_HT_bounds` | H_T bounds | Bounds |

## Key Definitions

| Name | Lean Identifier | Type | Module |
|------|-----------------|------|--------|
| Program | `Prog` | `Type u → Type u` | Programs |
| Feasible | `Feasible` | `Nat → Prog α → Prop` | Programs |
| MDL Cost | `mdlCost` | `FinDist α → Prog α → ℝ` | MDL |
| MDL∞ | `MDLinf` | `Nat → FinDist α → ℝ` | MDL |
| S_T | `ST` | `OptimalCondProg T PXY → Nat` | Core |
| H_T | `HT` | `OptimalCondProg T PXY → ℝ` | Core |
| STGap | `STGap` | `Nat → Nat → FinDist (α × β) → ℤ` | Emergence |
| Emergence | `EpiplexityEmergent` | `(∀ n, Nat → FinDist ...) → Prop` | Emergence |
| OWP | `OWP` | `(Nat → Type) → Prop` | Crypto/Axioms |
| PRF | `PRF` | `(Nat → Type) → Prop` | Crypto/Axioms |
| CSPRNG | `CSPRNG` | `(Nat → Type) → Prop` | Crypto/Axioms |

## Verification Status

All theorems verified with:
- `lake build --wfail` — No warnings as errors
- `guard_no_sorry.sh` — Zero `sorry` or `admit`
- Build date: 2026-01-15
