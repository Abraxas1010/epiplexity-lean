import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Real.Basic

/-!
Minimal finite-distribution core used by the Epiplexity paper pack.

This is intentionally small: we represent a distribution on a `Fintype` by a real-valued `pmf`,
with nonnegativity and total mass `1`.
-/

universe u

namespace HeytingLean
namespace Probability
namespace InfoTheory

noncomputable section

open scoped BigOperators

structure FinDist (α : Type u) [Fintype α] where
  pmf : α → ℝ
  nonneg : ∀ a, 0 ≤ pmf a
  sum_one : (∑ a : α, pmf a) = 1

namespace FinDist

variable {α : Type u} [Fintype α]

/-- Strict positivity of the mass function. -/
def Pos (P : FinDist α) : Prop :=
  ∀ a, 0 < P.pmf a

end FinDist

end

end InfoTheory
end Probability
end HeytingLean
