import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Real.Basic
import HeytingLean.Probability.InfoTheory.FinDist

universe u

namespace HeytingLean
namespace Epiplexity
namespace Crypto

open scoped BigOperators

noncomputable section

open HeytingLean.Probability.InfoTheory

namespace FinDist

/-- Pushforward of a finite distribution along a function. -/
noncomputable def map {α β : Type u} [Fintype α] [Fintype β] (f : α → β) (P : FinDist α) :
    FinDist β := by
  classical
  refine
    { pmf := fun b => ∑ a : α, if f a = b then P.pmf a else 0
      nonneg := ?_
      sum_one := ?_ }
  · intro b
    refine Finset.sum_nonneg ?_
    intro a _
    by_cases h : f a = b <;> simp [h, P.nonneg a]
  · -- Sum over all outputs, then swap sums.
    have hswap :
        (∑ b : β, ∑ a : α, if f a = b then P.pmf a else 0)
          = ∑ a : α, ∑ b : β, if f a = b then P.pmf a else 0 := by
      -- Both sides are the sum over `β × α`.
      have h1 :
          (∑ ba : β × α, if f ba.2 = ba.1 then P.pmf ba.2 else 0)
            = ∑ b : β, ∑ a : α, if f a = b then P.pmf a else 0 := by
        simpa using (Fintype.sum_prod_type
          (fun ba : β × α => if f ba.2 = ba.1 then P.pmf ba.2 else 0))
      have h2 :
          (∑ ba : β × α, if f ba.2 = ba.1 then P.pmf ba.2 else 0)
            = ∑ a : α, ∑ b : β, if f a = b then P.pmf a else 0 := by
        simpa using (Fintype.sum_prod_type_right
          (fun ba : β × α => if f ba.2 = ba.1 then P.pmf ba.2 else 0))
      exact h1.symm.trans h2
    calc
      (∑ b : β, ∑ a : α, if f a = b then P.pmf a else 0)
          = ∑ a : α, ∑ b : β, if f a = b then P.pmf a else 0 := hswap
      _ = ∑ a : α, P.pmf a := by
            refine Fintype.sum_congr
              (fun a : α => ∑ b : β, if f a = b then P.pmf a else 0)
              (fun a : α => P.pmf a)
              (fun a : α => ?_)
            simp
      _ = 1 := by
            simpa using P.sum_one

/-- Probability mass of a decidable event under a finite distribution. -/
noncomputable def probEvent {α : Type u} [Fintype α] (P : FinDist α) (E : α → Prop)
    [DecidablePred E] : ℝ :=
  ∑ a : α, if E a then P.pmf a else 0

end FinDist

/-- Probability that a Boolean test outputs `true` on samples from `P`. -/
noncomputable def probTrue {α : Type u} [Fintype α] (P : FinDist α) (D : α → Bool) : ℝ :=
  ∑ a : α, P.pmf a * (if D a then 1 else 0)

end
end Crypto
end Epiplexity
end HeytingLean
