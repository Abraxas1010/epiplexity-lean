import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
Core information-theoretic integrands for finite/discrete information theory.

We keep the definitions total by guarding `log` with `safeLog`.
-/

namespace HeytingLean
namespace Probability
namespace InfoTheory

noncomputable section

/-- Totalized logarithm: `log x` for `x > 0`, and `0` otherwise. -/
def safeLog (x : ℝ) : ℝ :=
  if 0 < x then Real.log x else 0

@[simp] lemma safeLog_of_pos {x : ℝ} (hx : 0 < x) : safeLog x = Real.log x := by
  simp [safeLog, hx]

@[simp] lemma safeLog_of_nonpos {x : ℝ} (hx : x ≤ 0) : safeLog x = 0 := by
  have : ¬ (0 < x) := not_lt_of_ge hx
  simp [safeLog, this]

/-- KL divergence integrand, totalized by a `p ≤ 0` guard. -/
def klTerm (p q : ℝ) : ℝ :=
  if p ≤ 0 then 0 else p * (safeLog p - safeLog q)

@[simp] lemma klTerm_of_nonpos {p q : ℝ} (hp : p ≤ 0) : klTerm p q = 0 := by
  simp [klTerm, hp]

/-- Shannon entropy integrand, totalized by a `p ≤ 0` guard. -/
def entropyTerm (p : ℝ) : ℝ :=
  if p ≤ 0 then 0 else (-p) * safeLog p

@[simp] lemma entropyTerm_of_nonpos {p : ℝ} (hp : p ≤ 0) : entropyTerm p = 0 := by
  simp [entropyTerm, hp]

end

end InfoTheory
end Probability
end HeytingLean

