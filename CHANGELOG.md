# Changelog

All notable changes to the Dim programming language compiler will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.5.0] - 2026-03-31

### Added
- **WASM Compilation**: `dim_wasm_codegen.py`
  - Compile Dim programs to WebAssembly (wasm32, wasm64 targets)
  - `runtime/dim_runtime_wasm.c`: WASM runtime with linear memory
- **LSP Server**: `dim_lsp.py`
  - Language Server Protocol implementation
  - Diagnostics, completions, hover, go-to-definition
  - Works with VS Code, Neovim, and other LSP clients
- **REPL**: `dim_repl.py`
  - Interactive Read-Eval-Print Loop
  - Command history, environment inspection
- **Debugger**: `dim_debugger.py`
  - Breakpoint management
  - Step-through debugging, variable inspection
- **Package Manager**: `dim_pkg.py`
  - Project initialization (`dim pkg init`)
  - Dependency management (`dim pkg add/remove`)
  - Registry support for package discovery
  - Local package caching
- **Build System**: `dim_build.py`
  - Project scaffolding (`dim new`)
  - Build targets (native, release, wasm)
  - Build and run (`dim run`)
- **Test Framework**: `dim_test.py`
  - Auto-discover test functions (`test_*`)
  - Run tests with filters
  - Test results reporting
- **Macro System**: `dim_macro.py`
  - Built-in macros: debug, todo, unimplemented
  - Macro expansion framework
- **Memory Management**:
  - Reference counting: `dim_alloc_ref`, `dim_inc_ref`, `dim_dec_ref`
  - Simple GC: `dim_gc_collect`, `dim_gc_register_root`
- **Concurrency**:
  - Thread pool: `dim_thread_pool_init`, `dim_thread_pool_submit`
  - Async/Futures: `dim_future_new`, `dim_future_await`, `dim_future_resolve`
- **Runtime Library**: `runtime/dim_runtime.c`
  - Math: sin, cos, tan, sqrt, pow, log, log10, exp, floor, ceil, round, fabs, PI, E
  - String: contains, starts_with, ends_with, index_of, replace, split, join, repeat
  - File I/O: read_file, write_file, file_exists, append
- **Standard Library Modules**:
  - `std/math.dim`: Full math functions
  - `std/str.dim`: String utilities
  - `std/file.dim`: File operations
  - `std/json.dim`: JSON parsing
- **New CLI Commands**:
  - `dim new <name>`: Create new project
  - `dim run [file]`: Build and run
  - `dim pkg <cmd>`: Package management
  - `dim bench`: Benchmark runner (stub)
- **Error Recovery**: Parser synchronization points
- **9 new tests** (70 total tests)

### Changed
- LLVM codegen: Extended runtime declarations for math, string, memory, concurrency
- CLI: Added new commands for pkg, run, new, bench
- Type checker: Added generics and traits support

### Fixed
- String method codegen: Proper RuntimeCallRValue handling
- Parser error recovery: Skip to statement boundary

## [0.4.0] - 2026-03-29

### Added
- **Runtime Library**: `runtime/dim_runtime.c`
  - Memory allocation: malloc, calloc, free, realloc
  - I/O operations: print_i32, print_str, println_str
  - String operations: len, concat, substring, to_upper, to_lower, trim
  - File I/O: read_file, write_file, file_exists
  - Math functions: abs_i32, min_i32, max_i32
- **Enum Variant Handling**: `Result.Ok(value)` syntax
  - `EnumVariant` AST node for enum variant construction
  - Parser support for `Enum.Variant` and `Enum.Variant(args)`
  - Type checking for enum variant arguments
- **String Methods**: `.upper()`, `.lower()`, `.trim()`, `.strip()`, `.split()`
- **Code Formatter**: `dim fmt` command
  - `dim_formatter.py`: Formats Dim source code
  - Configurable indent size
  - Consistent formatting for functions, structs, enums, etc.
- **4 new tests** (65 total tests)

### Changed
- LLVM codegen: Extended runtime declarations
- Type checker: Added string method type inference

## [0.4.0] - 2026-03-29

### Added
- **Module Import Resolution**: `import std.io`, `import std.vec`
  - `dim_module_resolver.py`: Module resolution and caching
  - Standard library modules in `std/` directory
  - `std/io.dim`: print, println, input, read_file, write_file, file_exists
  - `std/vec.dim`: push, pop, get, set, slice, reverse, sort, contains
- **4 new tests** (61 tests)

### Changed
- `TypeChecker` and `SemanticAnalyzer` now accept `module_resolver` parameter
- CLI commands updated to use module resolver

## [0.3.0] - 2026-03-29

### Added
- **Error Handling**: try/catch/finally and throw statements
- **Standard Library Functions**: len, range, assert, panic, abs, min, max
- **String Operations**: .len, .shape member access
- **Array/Tensor Operations**: .len, .shape member access
- **Closure Syntax**: `|x, y| -> expr` syntax with return type
- **Tuple Literals**: `(1, 2, 3)` syntax with type inference
- **Compound Assignment**: `+=`, `-=`, `*=`, `/=`, `%=` operators
- **Enhanced Loop Control**: break/continue with proper loop stack
- **Match Statement**: Pattern matching with MIR lowering
- **New Keywords**: try, catch, throw, finally, len, range, assert, panic

### Fixed
- Parser: Closure syntax with `|x, y| -> expr` 
- Type checker: PrimType import for unification
- Loop stack: Proper tracking for nested break/continue

### Changed
- Test suite expanded: 41 → 61 tests

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
