# Dim Compiler: Lowering Pipeline (MIR → MLIR → LLVM)

This document illustrates how Dim code is lowered through various IR levels to achieve high performance and hardware specialized execution.

## 1. Source Code (Dim)

```dim
fn add_tensors(a: Tensor[f32], b: f32) -> Tensor[f32]:
    return a + b
```

## 2. MIR (Dim Intermediate Representation)

At this level, the compiler performs ownership tracking and borrow checking.

```mir
func @add_tensors(%a: Buffer<f32>, %b: f32) -> Buffer<f32> {
    %1 = borrow %a : Buffer<f32>
    %2 = call @tensor_add_scalar(%1, %b)
    move_back %2 to result
    drop %a  # Ownership cleanup
    return result
}
```

## 3. MLIR (Multi-Level Intermediate Representation)

Dim uses the `Linalg` and `Tosa` dialects for tensor operations. This allows for kernel fusion.

```mlir
func @add_tensors(%arg0: tensor<?x?xf32>, %arg1: f32) -> tensor<?x?xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<?x?xf32>) outs(%arg0 : tensor<?x?xf32>) {
    ^bb0(%gen_arg: f32, %out: f32):
      %1 = arith.addf %gen_arg, %arg1 : f32
      linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
```

## 4. LLVM IR (Final Lowering)

The MLIR passes lower to standard LLVM IR for target-specific code generation (e.g., AArch64).

```llvm
define ptr @add_tensors(ptr %0, float %1) {
entry:
  %2 = load float, ptr %0
  %3 = fadd float %2, %1
  store float %3, ptr %res
  ret ptr %res
}
```

### Key Optimizations at Each Level:

- **MIR**: Borrow checking, dead-code elimination, move optimization.
- **MLIR**: Kernel fusion, loop tiling, vectorization (AVX/NEON), GPU lowering (CUDA).
- **LLVM**: Inline expansion, instruction selection, register allocation.
