# Dim Language Technical Specification (v0.1-draft)

## 1. Introduction

Dim is a statically-compiled, high-performance programming language designed to bridge the gap between Python's ergonomic developer experience and C++'s system-level control. It treats AI/ML and Security as first-class citizens, providing native constructs for LLM integration and hardware-accelerated tensor computations.

## 2. Language Core & Syntax

### 2.1 Grammar Overview

Dim uses an **indentation-based** layout. Significant whitespace (4 spaces) defines block scope, reducing visual noise from braces or semicolons.

```dim
fn calculate_risk(input: Tensor[f32], model: Model) -> f32:
    # Deterministic inference block
    with model.context:
        let prediction = model.forward(input)
        return prediction.mean()
```

### 2.2 Algebraic Type System & Ownership

Dim enforces a strict **Ownership & Borrowing** model inspired by Rust, but with ergonomic defaults.

- **Values & Moves**: Variables own their data. Assignment or function calls move ownership by default unless the type implements `Copy`.
- **References**:
  - `&T`: Immutable shared reference.
  - `&mut T`: Unique mutable reference. No other references can exist simultaneously.
- **Core ADTs**:
  - `Option[T]`: `Some(T) | None`
  - `Result[T, E]`: `Ok(T) | Err(E)`
- **Traits & Monomorphization**: Traits are resolved at compile-time for zero-cost abstraction. Optional dynamic dispatch (`dyn Trait`) is available for heterogeneous collections.

```dim
fn process_payload(owned_data: Buffer) -> Result[string, Error]:
    # owned_data is moved here
    let view = &owned_data # Shared borrow
    if view.is_valid():
        return Ok(view.to_string())
    return Err(Error.InvalidFormat)
```

### 2.3 Generics & Metaprogramming

Compile-time polymorphism using constraints.

```dim
trait Summable[T]:
    fn add(a: T, b: T) -> T

fn generic_add[T: Summable](x: T, y: T) -> T:
    return T.add(x, y)
```

## 3. Compiler Architecture

### 3.1 Pipeline

1.  **Lexer/Parser**: Greedy indentation tracking, PEG-based parsing for ambiguity resolution.
2.  **AST**: Minimalist Abstract Syntax Tree.
3.  **Semantic Analysis**: Scope resolution, symbol tables, and trait verification.
4.  **MIR (Dim Intermediate Representation)**: A control-flow graph (CFG) optimized for ownership tracking and borrow-checking.
5.  **LLVM IR / MLIR**:
    - **LLVM**: For generalized CPU targets (AArch64, x86_64).
    - **MLIR**: Specifically for tensor dialects and hardware-specific kernel lowering (SPIR-V, NVVM).

### 3.2 Compilation Modes

- **AOT**: Default mode for production binaries.
- **Incremental**: Hot-reloading support for fast dev cycles.
- **WASM**: Full support for browser/edge execution.
- **JIT**: Native JIT for dynamic tensor kernel fusion during training.

## 4. Memory & Execution Model

### 4.1 Hybrid Allocation Strategy

Dim utilizes a unified memory interface with three tiers:

1.  **Deterministic Ownership (Stack/Heap)**: Default mode where variables are tracked for lifetime. No manual `free`.
2.  **Region-Based Memory**: Grouped allocations for high-performance scoped workloads (e.g., frame-based game loops or request-handling in servers).
3.  **Optional GC**: A low-pause, concurrent garbage collector that can be enabled specifically for complex pointer graphs or web-interop layers.

### 4.2 Safety and `unsafe`

- **Memory Safety**: Out-of-bounds checks, null-safety (using `Optional[T]` variants), and data-race prevention are enforced at compile-time.
- **Unsafe Blocks**: Required for raw pointer manipulation, FFI, and performance-critical low-level hacks.

```dim
unsafe:
    let ptr = allocate_raw(1024)
    # Perform manual pointer arithmetic
```

## 5. Concurrency & Asynchrony

### 5.1 Structured Concurrency

Dim adopts a structured approach where lifetimes of child tasks are tied to their parents.

- **Actors**: Isolated state machines communicating via message passing.
- **Async/Await**: First-class support for non-blocking I/O, optimized for `io_uring` and `kqueue`.
- **Green Threads**: Lightweight M:N scheduling for high-concurrency servers.

```dim
fn crawl_site(url: string) async:
    spawn:
        let data = await fetch(url)
        process(data)
```

### 5.2 JavaScript Interop

When targeting WebAssembly, Dim's `async` maps directly to JS `Promises`, allowing seamless integration with browser event loops.

## 6. AI & LLM Integration (First-Class)

### 6.1 Native Prompt Objects & Structured Outputs

Prompts are first-class, type-checked objects that enforce structure on both inputs and outputs.

```dim
prompt BaseSystem:
    role system: "You are a secure coding assistant."

prompt AnalyzeSnippet(code: string) extends BaseSystem:
    role user: "Analyze this for vulnerabilities: {code}"
    # Verification: Compiler ensures the model output maps to this struct
    output: VulnerabilityReport

let report = await model.execute(AnalyzeSnippet("..."))
```

### 6.2 Tool Calling & Capability-Based Sandboxing

Functions exposed to AI models are isolated using a capability-based security model.

```dim
@tool(permissions=[NetRead, FileRead("/tmp")])
fn fetch_and_log(url: string) -> string:
    # Model can call this, but it can only read from /tmp
    return network.fetch(url)
```

- **Auditability**: Every tool execution is recorded in a cryptographically signed audit log.
- **Deterministic Inference**: Inference contexts are sandboxed, preventing models from leaking information between unrelated sessions.

## 7. Machine Learning Core

### 7.1 Native Tensor Types & Autodiff

Tensors are integrated into the type system with support for static shape verification.

```dim
fn training_step(x: Tensor[f32, [None, 784]], y: Tensor[i64]):
    let pred = model.forward(x)
    let loss = cross_entropy(pred, y)
    loss.backward() # Triggers the compiler-level gradient pass
```

- **Path-Based Autodiff**: The Dim compiler performs automatic differentiation on the MIR (Dim Intermediate Representation) before lowering, allowing for extreme optimizations like kernel fusion and dead-code elimination across the forward and backward passes.

### 7.2 MLIR-Based GPU Acceleration

The `dimc` compiler lowers high-level tensor operations to the `Linalg` and `TOSA` dialects of MLIR.

- **Kernel Fusion**: Dim glues together adjacent operations into single GPU kernels at compile-time.
- **Multi-Backend**: Native support for NVVM (CUDA), ROCDL (ROCm), and Metal (macOS) via MLIR's target-specific lowerings.

## 8. Cybersecurity & Systems Capabilities

### 8.1 Capability-Based Security & Taint Tracking

Dim enforces security at the language level by restricting access to system resources.

- **Capabilities**: Syscalls are only accessible through capability handles passed to functions.
- **Taint Tracking**: The compiler tracks data originating from untrusted sources (e.g., `NetSource`). Operations that mix tainted and untrusted data without sanitization trigger compile-time errors.

```dim
fn process_req(req: Request[Tainted]):
    let clean_data = sanitize(req.body)
    # Compiler error if sanitized is not called before using in SQL query
    db.query("SELECT * FROM users WHERE id = {clean_data}")
```

### 8.2 Formal Verification & Symbolic Execution

Dim includes first-class support for formal verification using an integrated SMT solver (Z3-based).

- **Verify Blocks**: Code within a `verify` block is symbolically executed at compile-time to prove properties like "no integer overflow" or "bounds are never exceeded".

```dim
verify:
    let x: u32 = ...
    let y: u32 = ...
    assert x + y >= x # Proves no overflow if configured
```

### 8.3 Secure Systems Modules

- **Binary DSL**: A declarative DSL for parsing ELF/PE formats with zero-copy and automatic bounds checking.
- **Audit Mode**: High-integrity logging that cannot be disabled by the process itself, enforced by the runtime/OS boundary.

## 9. Interop & ABI

### 9.1 Stable C ABI

Dim defines a stable C-compatible ABI for exported functions, facilitating seamless calling from C, C++, Rust, or Zig.

### 9.2 JS/TS Bidirectional FFI

- **Importing JS**: Use `extern "js"` to import JS functions with automatic type conversion.
- **Exporting to JS**: Dim modules compile to ES modules when targeting WASM.

## 10. Tooling & Ecosystem

### 10.1 `dim` CLI

The unified tool for everything:

- `dim build`: Incremental AOT compilation.
- `dim run`: Run script or binary.
- `dim test`: Built-in test runner with Fuzzing capabilities.
- `dim pkg`: Package manager with cryptographic dependency verification.

### 10.2 Developer Experience

- **Formatting**: `dim fmt` (opinionated, zero-config).
- **Linter**: `dim lint` includes security-focused rules (e.g., checking for raw syscalls in non-system modules).
- **AI-Assisted Debugging**: The debugger integrates with local LLMs to explain crash dumps and suggest fixes based on the MIR state.

## 11. Conclusion

Dim is designed for the next generation of software where performance, safety, and AI are inseparable. By integrating high-level ergonomics with low-level control and first-class AI constructs, Dim provides a unique platform for building secure, high-performance intelligent systems.
