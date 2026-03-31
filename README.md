# Dim Programming Language

**Dim** is a next-generation, statically-compiled programming language designed to bridge ergonomic developer experience with systems-level control — with **AI/ML**, **memory safety**, and **security** as first-class language features.

> **Status:** v0.5.0 Production — Full compiler pipeline, stdlib, LSP, REPL, debugger, package manager

---

## Features

| Feature                     | Status                                         |
| --------------------------- | ---------------------------------------------- |
| Lexer with INDENT/DEDENT    | ✅ Production (`dim_lexer.py`)                 |
| EBNF Grammar                | ✅ Formal spec (`dim_grammar.ebnf`)            |
| Span-annotated AST          | ✅ v0.5 (`dim_ast.py`)                         |
| Algebraic Type System       | ✅ (`dim_types.py`)                            |
| Hindley-Milner Type Checker | ✅ (`dim_type_checker.py`)                     |
| Structured Diagnostics      | ✅ (`dim_diagnostic.py`)                       |
| Mid-Level IR (MIR / CFG)    | ✅ (`dim_mir.py`)                              |
| AST → MIR Lowering          | ✅ (`dim_mir_lowering.py`)                     |
| Borrow Checker (MIR-level)  | ✅ Polonius-inspired (`dim_borrow_checker.py`) |
| LLVM IR Codegen             | ✅ (`dim_mir_to_llvm.py`)                      |
| Module Import Resolution    | ✅ (`dim_module_resolver.py`)                  |
| First-class `prompt` type   | ✅ AST + type system                           |
| `actor` / message-passing   | ✅ AST + parser                                |
| `@tool` decorator           | ✅ Parser + semantic analysis                  |
| Error handling (try/catch) | ✅ Parser + type checker                       |
| Closures & lambdas          | ✅ `|x, y| -> expr` syntax                     |
| Generics & Traits           | ✅ (`dim_type_checker.py`, `dim_types.py`)    |
| FFI (foreign/use)           | ✅ Parser + type checker                       |
| Standard library            | ✅ Full working stdlib                         |
| WASM compilation            | ✅ (`dim_wasm_codegen.py`)                     |
| Package Manager             | ✅ (`dim_pkg.py`)                              |
| LSP Server                  | ✅ (`dim_lsp.py`)                              |
| REPL                        | ✅ (`dim_repl.py`)                             |
| Debugger                    | ✅ (`dim_debugger.py`)                         |
| Test Framework              | ✅ (`dim_test.py`)                            |
| Build System                | ✅ (`dim_build.py`)                            |
| Macro System                | ✅ (`dim_macro.py`)                            |
| Memory Management (RC + GC)| ✅ Reference counting + GC                    |
| Concurrency (Threads/Future)| ✅ Thread pool + async/futures                |
| Test Suite                  | ✅ 70 tests (`dim_tests.py`)                   |

---

## Syntax Preview

```dim
# Functions with return types
fn add(x: i32, y: i32) -> i32:
    return x + y

# Ownership — immutable by default, move semantics
fn process(data: Buffer) -> Result[str, Error]:
    let view = &data          # shared borrow
    return Ok(view.to_string())

# Async / Await
async fn fetch(url: String) -> Result[Data, Error]:
    let resp = await http::get(url)
    return parse(resp.body)

# Actors — isolated state via message passing
actor Counter:
    state: i32 = 0
    receive Increment():
        self.state += 1
    receive GetCount() -> i32:
        return self.state

# AI as a first-class type
prompt Classify:
    role system: "You are a text classifier."
    role user:   "Classify: {input}"
    output: ClassLabel

# Pattern matching
fn describe(x: i32) -> str:
    match x:
        0: return "zero"
        n if n > 0: return "positive"
        _: return "negative"

# Generics + Traits
trait Summable[T]:
    fn add(a: T, b: T) -> T

fn generic_sum[T: Summable](a: T, b: T) -> T:
    return T.add(a, b)

# Error handling
fn safe_div(a: i32, b: i32):
    try:
        if b == 0:
            throw Error()
        return a / b
    catch e:
        print("Error!")
    finally:
        cleanup()

# Closures and tuples
fn use_closure():
    let add = |x, y| -> x + y
    let point = (1, 2, 3)
    let len = point.len

# Standard library
fn demo_std():
    let x = abs(-5)
    let m = min(1, 2)
    let mx = max(1, 2)
    let s = sin(3.14159)
    let root = sqrt(144.0)

# Module imports
import std.io
import std.vec
import std.math
import std.str

fn demo_io():
    print("Hello, World!")
    let content = read_file("test.txt")
    if file_exists("test.txt"):
        println("File exists!")

fn demo_vec():
    let arr = [1, 2, 3]
    push(arr, 4)
    let x = pop(arr)

# FFI - call C functions
foreign "libc.so" [
    fn puts(msg: str) -> i32
    fn rand() -> i32
]
```

---

## Compiler Pipeline

```
Source (.dim)
    ↓  dim_lexer.py           — Tokenisation + INDENT/DEDENT
    ↓  dim_parser.py          — AST construction (Pratt + recursive descent)
    ↓  dim_module_resolver.py — Module import resolution
    ↓  dim_type_checker.py    — HM type inference, scope resolution
    ↓  dim_mir_lowering.py   — AST → MIR (SSA Control Flow Graph)
    ↓  dim_borrow_checker.py — Ownership & borrow validation (Polonius)
    ↓  dim_mir_to_llvm.py     — MIR → LLVM IR (x86_64, wasm32, wasm64)
    ↓  clang                  — Native binary / WASM
```

---

## Usage

```bash
# Initialize a new project
dim new myproject
cd myproject

# Build and run
dim run main.dim
dim run main.dim arg1 arg2

# Just type-check
dim check hello.dim

# Print AST
dim parse hello.dim

# Print MIR (SSA/CFG)
dim mir hello.dim

# Run borrow checker
dim borrow hello.dim

# Run full build pipeline
dim build hello.dim

# Run test suite
dim test
dim test --tag lexer   # filter by category

# Package manager
dim pkg init mypkg
dim pkg add http 1.0.0
dim pkg remove http
dim pkg install
dim pkg list

# Format code
dim fmt hello.dim

# LSP server (for IDEs)
python dim_lsp.py

# REPL
python dim_repl.py

# Debugger
python dim_debugger.py hello.dim
```

---

## Project Structure

```
dim/                      — Dim source root
├── Compiler Core
│   ├── dim_token.py          — Token and Span definitions
│   ├── dim_lexer.py          — Lexer with INDENT/DEDENT
│   ├── dim_parser.py         — Recursive descent parser
│   ├── dim_ast.py            — AST node definitions
│   ├── dim_types.py          — Type system
│   ├── dim_type_checker.py   — Hindley-Milner type inference
│   ├── dim_semantic.py       — Semantic analysis
│   ├── dim_borrow_checker.py — Borrow checking
│   └── dim_module_resolver.py — Module resolution
├── IR & Codegen
│   ├── dim_mir.py            — Mid-Level IR
│   ├── dim_mir_lowering.py   — AST → MIR lowering
│   ├── dim_mir_to_llvm.py    — MIR → LLVM IR
│   ├── dim_native_codegen.py — Native binary generation
│   └── dim_wasm_codegen.py   — WASM compilation
├── Tools
│   ├── dim_cli.py            — CLI interface
│   ├── dim_diagnostic.py     — Error/warning system
│   ├── dim_lsp.py            — LSP server (IDE support)
│   ├── dim_repl.py           — Interactive REPL
│   ├── dim_debugger.py       — Debugger
│   ├── dim_test.py           — Test framework
│   ├── dim_pkg.py            — Package manager
│   ├── dim_build.py          — Build system
│   ├── dim_formatter.py      — Code formatter
│   └── dim_macro.py         — Macro system
├── Runtime
│   ├── runtime/dim_runtime.c         — C runtime (native)
│   └── runtime/dim_runtime_wasm.c   — C runtime (WASM)
├── Stdlib
│   ├── std/io.dim    — I/O utilities
│   ├── std/vec.dim   — Vector/array utilities
│   ├── std/math.dim  — Math functions
│   ├── std/str.dim   — String utilities
│   ├── std/file.dim  — File I/O
│   └── std/json.dim  — JSON utilities
├── Tests & Docs
│   ├── dim_tests.py          — 70 test cases
│   └── examples/             — Example .dim files
│       ├── functions.dim
│       ├── control_flow.dim
│       ├── ai_tools.dim
│       ├── types.dim
│       ├── ownership.dim
│       ├── traits.dim
│       ├── ffi.dim
│       └── test_example.dim
└── Documentation
    ├── README.md             — This file
    ├── CHANGELOG.md          — Version history
    ├── CONTRIBUTING.md       — Contribution guide
    ├── LICENSE.md            — MIT License
    ├── COMPILER_INTERNALS.md — Compiler internals
    ├── QUICK_REFERENCE.md    — Language quick reference
    └── dim_specification.md   — Language specification
```

---

## Roadmap

| Phase | Milestone                                                | Status |
| ----- | -------------------------------------------------------- | ------ |
| 0     | Lexer, Parser, AST prototype                             | ✅ Done |
| 1     | Types, MIR, Borrow Checker, Diagnostics                  | ✅ Done |
| 2     | LLVM IR codegen, function calls, @tool parsing            | ✅ Done |
| 3     | Native binaries, WASM, full LLVM backend                  | ✅ Done |
| 4     | Package manager, LSP, REPL, Debugger                      | ✅ Done |
| 5     | AI/LLM engine (typed prompts, model adapters)            | 🔜     |
| 6     | Security: taint analysis, capability model, Z3 contracts | 🔜     |
| 7     | Tooling: linter, benchmark, documentation generator       | 🔜     |
| 8     | Self-hosting, v1.0                                       | 🔜     |

---

## Design Goals

- **Memory safe** without garbage collection — ownership + borrow checker
- **AI-native** — `prompt` and `model` are language keywords, not library calls
- **Secure by default** — capability-based access, static taint analysis
- **Ergonomic** — Python-like syntax, indentation-based blocks, HM type inference
- **High performance** — compiles to native code via LLVM
- **Web-ready** — WASM compilation for browser/serverless

---

_Dim — where systems programming meets intelligence._