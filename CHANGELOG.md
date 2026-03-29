# Changelog

All notable changes to the Dim programming language compiler will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.0] - 2026-03-29

### Added
- **LLVM IR Codegen** (`dim_mir_to_llvm.py`): Full MIR to LLVM IR translation for x86_64
  - Function declarations and definitions
  - Binary operations (add, sub, mul, div, comparison)
  - Function calls with proper result assignment
  - Branch and goto terminators
  - Return statements with void/unit support
- **@tool Decorator**: Capability-based security model for AI tool calling
  - `permissions=[Cap1, Cap2('/path')]` syntax
  - Semantic validation of capabilities
- **11 new tests** (30 → 41 tests)
  - @tool decorator parsing (3 tests)
  - MIR if/else and while loop (2 tests)
  - LLVM codegen (2 tests)
  - Type inference and binary ops (2 tests)

### Fixed
- Function call MIR lowering: Call terminator now correctly creates continuation blocks
- `_type_to_llvm`: `UNIT` type now maps to `void` instead of `i8*`
- Return terminator: `ret void` without value for Unit-returning functions
- `_parse_function`: Skip newlines before expecting `fn` keyword
- `_parse_decorated_function`: Full `permissions=[...]` syntax support
- `_block_names`: Uses Python `id(bb)` to avoid cross-function block ID collisions

### Changed
- Test suite expanded: 30 → 35 tests
- Project status: Phase 1 → Phase 2

## [0.1.0] - 2026-02-18

### Added
- **Lexer** (`dim_lexer.py`): Production tokenizer with INDENT/DEDENT handling
- **Parser** (`dim_parser.py`): Recursive descent + Pratt expression parser
- **AST** (`dim_ast.py`): Span-annotated abstract syntax tree
- **Type System** (`dim_types.py`): Algebraic types (primitives, generics, tensors, prompts)
- **Type Checker** (`dim_type_checker.py`): Hindley-Milner type inference
- **MIR** (`dim_mir.py`): Mid-level IR with SSA locals and BasicBlocks
- **MIR Lowering** (`dim_mir_lowering.py`): AST → MIR translation
- **Borrow Checker** (`dim_borrow_checker.py`): Polonius-inspired ownership validation
- **Semantic Analyzer** (`dim_semantic.py`): Scope resolution and trait verification
- **Diagnostics** (`dim_diagnostic.py`): Structured error/warning system
- **CLI** (`dim_cli.py`): Unified compiler command-line interface
- **Test Suite** (`dim_tests.py`): 30 test cases covering lexer, parser, type checker, MIR, borrow checker
- **EBNF Grammar** (`dim_grammar.ebnf`): Formal language specification

### Features
- Indentation-based syntax (Python-like)
- Algebraic data types (structs, enums, traits)
- Pattern matching (match statements)
- Async/await support
- First-class `prompt` type for AI/LLM integration
- Actor model for message-passing concurrency
- Borrow expressions (`&`, `&mut`)
- Generics with trait constraints
