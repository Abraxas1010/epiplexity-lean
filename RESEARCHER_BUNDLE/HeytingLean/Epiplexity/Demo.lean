import HeytingLean

/-!
Minimal executable entrypoint used by the `EpiplexityBundle` paper pack.

The `#check` commands force typechecking of the main API/results during compilation.
-/

open HeytingLean.Epiplexity

#check BitStr.crossEntropyBits_uniform
#check Crypto.CSPRNG.theorem17
#check Crypto.Factorization.condCrossEntropyBits_jointYX_ge_log2_invSuccess

def main : IO Unit := do
  IO.println "epiplexity_demo: ok"
