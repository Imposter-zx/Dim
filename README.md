# Dim Programming Language

**Dim** is a next-generation, statically-compiled programming language designed to bridge ergonomic developer experience with systems-level control — with **AI/ML**, **memory safety**, and **security** as first-class language features.

> **Status:** Phase 2 Foundation (compiler frontend + MIR + borrow checker + LLVM codegen)

---

## Features

| Feature                     | Status                                         |
| --------------------------- | ---------------------------------------------- |
| Lexer with INDENT/DEDENT    | ✅ Production (`dim_lexer.py`)                 |
| EBNF Grammar                | ✅ Formal spec (`dim_grammar.ebnf`)            |
| Span-annotated AST          | ✅ v0.2 (`dim_ast.py`)                         |
| Algebraic Type System       | ✅ (`dim_types.py`)                            |
| Hindley-Milner Type Checker | ✅ (`dim_type_checker.py`)                     |
| Structured Diagnostics      | ✅ (`dim_diagnostic.py`)                       |
| Mid-Level IR (MIR / CFG)    | ✅ (`dim_mir.py`)                              |
| AST → MIR Lowering          | ✅ (`dim_mir_lowering.py`)                     |
| Borrow Checker (MIR-level)  | ✅ Polonius-inspired (`dim_borrow_checker.py`) |
| LLVM IR Codegen             | ✅ (`dim_mir_to_llvm.py`)                      |
| First-class `prompt` type   | ✅ AST + type system                           |
| `actor` / message-passing   | ✅ AST + parser                                |
| `@tool` decorator           | ✅ Parser + semantic analysis                   |
| Error handling (try/catch) | ✅ Parser + type checker                       |
| Standard library            | ✅ len, abs, min, max, assert, panic         |
| Closures & lambdas          | ✅ `|x, y| -> expr` syntax                   |
| Test Suite                  | ✅ 54 tests (`dim_tests.py`)                   |
| Native binaries + WASM      | 🔜 Phase 3                                     |
| Async runtime               | 🔜 Phase 4                                     |
| Package manager             | 🔜 Phase 7                                     |
| LSP server                  | 🔜 Phase 7                                     |

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
    assert x > 0
```

---

## Compiler Pipeline

```
Source (.dim)
    ↓  dim_lexer.py          — Tokenisation + INDENT/DEDENT
    ↓  dim_parser.py         — AST construction (Pratt + recursive descent)
    ↓  dim_type_checker.py   — HM type inference, scope resolution
    ↓  dim_mir_lowering.py   — AST → MIR (SSA Control Flow Graph)
    ↓  dim_borrow_checker.py — Ownership & borrow validation (Polonius)
    ↓  dim_mir_to_llvm.py    — MIR → LLVM IR (x86_64)
```

---

## Usage

```bash
# Type-check a file
python dim_cli.py check hello.dim

# Print AST
python dim_cli.py parse hello.dim

# Print MIR (SSA/CFG)
python dim_cli.py mir hello.dim

# Run borrow checker
python dim_cli.py borrow hello.dim

# Run full build pipeline
python dim_cli.py build hello.dim

# Run test suite
python dim_cli.py test
python dim_cli.py test --tag lexer   # filter by category
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
│   └── dim_semantic.py       — Semantic analysis
├── IR & Codegen
│   ├── dim_mir.py            — Mid-Level IR
│   ├── dim_mir_lowering.py   — AST → MIR lowering
│   ├── dim_mir_to_llvm.py    — MIR → LLVM IR
│   └── dim_borrow_checker.py — Borrow checking
├── Tools
│   ├── dim_diagnostic.py     — Error/warning system
│   └── dim_cli.py            — CLI interface
├── Tests & Docs
│   ├── dim_tests.py          — 41 test cases
│   ├── test.dim              — Sample source
│   └── examples/             — Example .dim files
│       ├── functions.dim
│       ├── control_flow.dim
│       ├── ai_tools.dim
│       ├── types.dim
│       └── ownership.dim
├── Documentation
│   ├── README.md             — This file
│   ├── CHANGELOG.md         — Version history
│   ├── CONTRIBUTING.md       — Contribution guide
│   ├── LICENSE.md            — MIT License
│   ├── COMPILER_INTERNALS.md — Compiler internals
│   ├── QUICK_REFERENCE.md   — Language quick reference
│   └── dim_specification.md  — Language specification
└── dim_grammar.ebnf          — Formal EBNF grammar
```

---

## Roadmap

| Phase | Milestone                                                | Target                 |
| ----- | -------------------------------------------------------- | ---------------------- |
| 0     | Lexer, Parser, AST prototype                             | ✅ Done                |
| 1     | Types, MIR, Borrow Checker, Diagnostics                  | ✅ Done                |
| 2     | LLVM IR codegen, function calls, @tool parsing           | ✅ Done (this release) |
| 3     | Native binaries, WASM, full LLVM backend                  | Q4 2026                |
| 4     | Async runtime, actor scheduler                           | Q1 2027                |
| 5     | AI/LLM engine (typed prompts, model adapters)            | Q2 2027                |
| 6     | Security: taint analysis, capability model, Z3 contracts | Q3 2027                |
| 7     | Tooling: formatter, linter, package manager, LSP         | Q4 2027                |
| 8     | Self-hosting, v1.0                                       | 2028                   |

---

## Design Goals

- **Memory safe** without garbage collection — ownership + borrow checker
- **AI-native** — `prompt` and `model` are language keywords, not library calls
- **Secure by default** — capability-based access, static taint analysis
- **Ergonomic** — Python-like syntax, indentation-based blocks, HM type inference
- **High performance** — compiles to native code via LLVM

---

_Dim — where systems programming meets intelligence._
