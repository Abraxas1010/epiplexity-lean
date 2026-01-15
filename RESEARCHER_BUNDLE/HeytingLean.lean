import HeytingLean.Epiplexity.Prelude
import HeytingLean.Epiplexity.Programs
import HeytingLean.Epiplexity.Info
import HeytingLean.Epiplexity.MDL
import HeytingLean.Epiplexity.Core
import HeytingLean.Epiplexity.Bounds
import HeytingLean.Epiplexity.Conditional
import HeytingLean.Epiplexity.Emergence
import HeytingLean.Epiplexity.Crypto.Axioms
import HeytingLean.Epiplexity.Crypto.CSPRNG
import HeytingLean.Epiplexity.Crypto.HeavySet
import HeytingLean.Epiplexity.Crypto.PRFHighEpiplexity
import HeytingLean.Epiplexity.Crypto.Factorization

/-!
# Epiplexity (umbrella)

Lean 4 formalization of core definitions from "From Entropy to Epiplexity" (Finzi et al., 2026):

- MDL-style epiplexity `S_T` and time-bounded entropy `H_T`,
- conditional epiplexity / time-bounded conditional entropy,
- epiplexity-emergence predicate,
- cryptographic hypothesis predicates and the heavy-set lemmas used in the paper's bounds.

## Main Results

- **Theorem 9**: CSPRNG → High Epiplexity
- **Theorem 12**: CSPRNGβ → β-conditional high epiplexity
- **Theorem 13**: Factorization hardness under OWP
- **Theorem 17-19**: CSPRNG characterizations
- **Theorem 24**: PRF High Epiplexity
- **Theorem 25**: OWP Factorization Hardness (main)
- **Corollary 26**: OWP average-case Factorization
-/
