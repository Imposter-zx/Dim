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
| Test Suite                  | ✅ 35 tests (`dim_tests.py`)                   |
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
dim_token.py           — Token dataclass + Span source locations
dim_lexer.py           — Production lexer (replaces dim_poc_lexer.py)
dim_ast.py             — Span-annotated AST node definitions
dim_parser.py          — Recursive descent + Pratt expression parser
dim_types.py           — Algebraic type system (primitives, generics, tensors, prompts)
dim_type_checker.py    — Hindley-Milner type inference engine
dim_diagnostic.py       — Structured error/warning system with source highlighting
dim_mir.py             — Mid-Level IR: SSA locals, BasicBlocks, CFG algorithms
dim_mir_lowering.py    — AST → MIR lowering pass
dim_mir_to_llvm.py     — MIR → LLVM IR codegen (x86_64)
dim_borrow_checker.py  — Ownership & borrow checking (Polonius-inspired)
dim_semantic.py        — Top-level semantic analysis orchestrator
dim_cli.py             — Unified compiler CLI
dim_tests.py           — Test suite (35 test cases)
dim_grammar.ebnf       — Formal EBNF grammar specification
dim_specification.md   — Full language technical specification
test.dim               — Sample Dim source file
```
dim_token.py           — Token dataclass + Span source locations
dim_lexer.py           — Production lexer (replaces dim_poc_lexer.py)
dim_ast.py             — Span-annotated AST node definitions
dim_parser.py          — Recursive descent + Pratt expression parser
dim_types.py           — Algebraic type system (primitives, generics, tensors, prompts)
dim_type_checker.py    — Hindley-Milner type inference engine
dim_diagnostic.py      — Structured error/warning system with source highlighting
dim_mir.py             — Mid-Level IR: SSA locals, BasicBlocks, CFG algorithms
dim_mir_lowering.py    — AST → MIR lowering pass
dim_borrow_checker.py  — Ownership & borrow checking (Polonius-inspired)
dim_semantic.py        — Top-level semantic analysis orchestrator
dim_cli.py             — Unified compiler CLI
dim_tests.py           — Test suite (30 test cases)
dim_grammar.ebnf       — Formal EBNF grammar specification
dim_specification.md   — Full language technical specification
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
