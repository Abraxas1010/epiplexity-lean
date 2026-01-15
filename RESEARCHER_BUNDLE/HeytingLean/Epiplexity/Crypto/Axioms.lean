import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Real.Basic
import HeytingLean.Epiplexity.Crypto.FinDistExtras
import HeytingLean.Epiplexity.Prelude

universe u

namespace HeytingLean
namespace Epiplexity
namespace Crypto

open scoped BigOperators

noncomputable section

open HeytingLean.Probability.InfoTheory

/-!
Security predicates (assumption layer).

These are *Props* packaging the usual computational indistinguishability-style hypotheses on finite
probability spaces, without committing to a particular PPT/universal-machine model.
-/

/-- Distinguishing advantage between distributions `P` and `Q` for a Boolean test. -/
noncomputable def advantage {α : Type u} [Fintype α] (P Q : FinDist α) (D : α → Bool) : ℝ :=
  |probTrue P D - probTrue Q D|

/-- A minimal distinguisher wrapper (tracked only by a time budget). -/
structure Distinguisher (α : Type u) where
  run : α → Bool
  runtime : Nat

namespace Distinguisher

instance {α : Type u} : CoeFun (Distinguisher α) (fun _ => α → Bool) :=
  ⟨fun D => D.run⟩

end Distinguisher

/-- Definition 3 (CSPRNG), as a finite indistinguishability predicate with advantage bound `ε(k)`. -/
def CSPRNGSecure (k n T : Nat) (G : BitStr k → BitStr n) (ε : Nat → ℝ) : Prop :=
  ∀ D : Distinguisher (BitStr n),
    D.runtime ≤ T →
      advantage
          (FinDist.map G (Epiplexity.FinDist.uniform (α := BitStr k)))
          (Epiplexity.FinDist.uniform (α := BitStr n))
          D.run
        ≤ ε k

/-- A minimal inverter wrapper (tracked only by a time budget). -/
structure Inverter (α β : Type u) where
  run : β → α
  runtime : Nat

namespace Inverter

instance {α β : Type u} : CoeFun (Inverter α β) (fun _ => β → α) :=
  ⟨fun A => A.run⟩

end Inverter

/-- Success probability of an inverter against `f`, sampling `x` uniformly. -/
noncomputable def owfSuccess {α β : Type u} [Fintype α] [Nonempty α] [DecidableEq β]
    (f : α → β) (A : Inverter α β) : ℝ :=
  let U : FinDist α := Epiplexity.FinDist.uniform (α := α)
  ∑ x : α, U.pmf x * (if f (A.run (f x)) = f x then 1 else 0)

/-- Definition 4 (OWF), as a finite inversion-success bound `≤ ε(n)` under a time budget. -/
def OWFSecure (α β : Type u) [Fintype α] [Nonempty α] [Fintype β] [DecidableEq β]
    (T : Nat) (f : α → β) (ε : Nat → ℝ) : Prop :=
  ∀ A : Inverter α β, A.runtime ≤ T → owfSuccess (f := f) A ≤ ε (Fintype.card α)

/-- Definition 20 (PRF), modeled as indistinguishability of a random keyed function vs a uniform random function. -/
def PRFSecure (k n m T : Nat) (F : BitStr k → BitStr n → BitStr m) (ε : Nat → ℝ) : Prop :=
  ∀ D : Distinguisher (BitStr n → BitStr m),
    D.runtime ≤ T →
      advantage
          (FinDist.map (fun key => F key) (Epiplexity.FinDist.uniform (α := BitStr k)))
          (Epiplexity.FinDist.uniform (α := (BitStr n → BitStr m)))
          D.run
        ≤ ε k

end

end Crypto
end Epiplexity
end HeytingLean
