import Mathlib.Data.List.Basic

/-!
Minimal Algorithmic Information Theory (AIT) shim for the Epiplexity paper pack.

`Program` is represented as a finite bitstring (`List Bool`), with `codeLength` as its length.
-/

namespace HeytingLean.Meta.AIT

/-- A program is represented as a finite bitstring. -/
abbrev Program := List Bool

namespace Program

/-- The length of a program (number of bits). -/
def length (p : Program) : Nat :=
  List.length (p : List Bool)

@[simp] lemma length_nil : length ([] : Program) = 0 := by
  simp [length]

@[simp] lemma length_cons (b : Bool) (p : Program) :
    length (b :: p) = length p + 1 := by
  simp [length]

end Program

/-- Length of a program, exposed at the AIT level. -/
def codeLength (p : Program) : Nat :=
  Program.length p

@[simp] lemma codeLength_nil : codeLength ([] : Program) = 0 := by
  simp [codeLength, Program.length]

@[simp] lemma codeLength_cons (b : Bool) (p : Program) :
    codeLength (b :: p) = codeLength p + 1 := by
  simp [codeLength, Program.length_cons]

end HeytingLean.Meta.AIT

