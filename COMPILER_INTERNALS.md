# Dim Compiler Internals

This document provides an in-depth look at the Dim compiler's architecture and implementation details.

## Overview

The Dim compiler is a multi-phase compiler written in Python 3.6.8+. It follows a traditional compiler pipeline:

```
Source (.dim)
    ↓ Lexer (dim_lexer.py)
    ↓ Parser (dim_parser.py)
    ↓ AST
    ↓ Semantic Analysis (dim_semantic.py)
    ↓ Type Checker (dim_type_checker.py)
    ↓ MIR (dim_mir_lowering.py)
    ↓ Borrow Checker (dim_borrow_checker.py)
    ↓ LLVM IR (dim_mir_to_llvm.py)
```

## Phase 1: Lexer

**File:** `dim_lexer.py`

The lexer converts source code into a stream of tokens. Key features:

- **Indentation tracking**: Injects `INDENT`/`DEDENT` tokens for Python-style blocks
- **Keyword recognition**: Distinguishes keywords from identifiers
- **Span tracking**: Every token includes source location information

### Token Types

Key token types in `dim_token.py`:
- `IDENTIFIER` - Variable/function names
- `KEYWORD` - Language keywords (fn, let, if, etc.)
- `INTEGER`, `FLOAT`, `STRING` - Literals
- `INDENT`, `DEDENT` - Block structure
- `AT` - Decorators (@)

## Phase 2: Parser

**File:** `dim_parser.py`

Uses recursive descent parsing with Pratt expression parsing for precedence:

### Expression Parsing

Uses the Pratt parser pattern for expressions:
```
_parse_expression() → _parse_binary() → _parse_unary() → _parse_postfix() → _parse_primary()
```

### Statement Parsing

Top-level statements are parsed based on keywords:
- `fn`/`async` → `_parse_function()`
- `@` → `_parse_decorated_function()`
- `let` → `_parse_let()`
- `if` → `_parse_if()`
- etc.

### AST Nodes

Defined in `dim_ast.py`:
- `FunctionDef` - Function definitions
- `LetStmt` - Variable bindings
- `ReturnStmt` - Return statements
- `IfStmt` - Conditional statements
- `BinaryOp`, `UnaryOp` - Operations
- `Call` - Function calls
- `BorrowExpr` - Borrow expressions
- etc.

## Phase 3: Type System

**File:** `dim_types.py`

### Type Hierarchy

```
Type (base)
├── PrimType (i32, f64, bool, string, unit)
├── RefType (&T, &mut T)
├── TensorType (Tensor[T, shape])
├── PromptType (prompt definitions)
├── FnType (function types)
├── GenericType (generic parameters)
└── TypeVar (for inference)
```

### Unification

The type checker uses unification for type inference:
- Match concrete types directly
- Instantiate type variables with concrete types
- Propagate constraints through expressions

## Phase 4: Type Checker

**File:** `dim_type_checker.py`

Implements Hindley-Milner type inference with extensions for Dim-specific features.

### Type Inference Rules

1. **Literals**: Infer from value (42 → i32, 3.14 → f32)
2. **Variables**: Look up declared type
3. **Let bindings**: Infer from expression, constrain to annotation
4. **Functions**: Infer return type from body
5. **Borrows**: Create RefType wrapper

### Error Codes

- `E0020`: Undefined variable
- `E0021`: Undefined function
- `E0030`: Type mismatch
- `E0040`: Use of moved value
- `E0041`: Double mutable borrow

## Phase 5: MIR Lowering

**File:** `dim_mir_lowering.py`

Converts AST to Mid-Level IR (MIR) - a Control Flow Graph (CFG) representation.

### MIR Structure

```
MIRModule
└── MIRFunction
    ├── params: List[Local]
    ├── locals: List[Local]
    ├── return_type: Type
    └── blocks: List[BasicBlock]
        ├── stmts: List[MIRStatement]
        └── terminator: Terminator
```

### MIR Statements

- `StorageLive(local)` - Mark local as live
- `StorageDead(local)` - Mark local as dead
- `Assign(place, rvalue)` - Assignment
- `Borrow(dest, place)` - Create borrow
- `Drop(place)` - Drop value

### Terminators

- `Return(value)` - Return from function
- `Goto(target)` - Unconditional jump
- `Branch(condition, true, false)` - Conditional branch
- `Call(callee, args, dest, next_block)` - Function call

## Phase 6: Borrow Checker

**File:** `dim_borrow_checker.py`

Implements Polonius-inspired borrow checking using loan lifetime analysis.

### Key Concepts

- **Loans**: Each borrow creates a loan with a lifetime
- **Paths**: Dereferences and field accesses create paths
- **Violations**: Detects use-after-move, double mutable borrow

### Liveness Analysis

Used for precise borrow checking:
- Compute which locals are live at each point
- Loans outlive their borrowers
- Invalidates loans on move

## Phase 7: LLVM Codegen

**File:** `dim_mir_to_llvm.py`

Generates LLVM IR from MIR.

### Type Mapping

| Dim Type | LLVM Type |
|----------|-----------|
| i32 | i32 |
| i64 | i64 |
| f32 | float |
| f64 | double |
| bool | i1 |
| string | i8* |
| Unit | void |
| Tensor[T, n] | <n x float> |

### Function Calls

Function calls generate:
1. `call` instruction with result
2. `br` to continuation block

Example:
```llvm
%result = call i32 @add(i32 %a, i32 %b)
br label %bb1
```

## CLI Interface

**File:** `dim_cli.py`

### Commands

- `dim lex <file>` - Lex only
- `dim parse <file>` - Parse and print AST
- `dim check <file>` - Type check
- `dim mir <file>` - Lower to MIR and print
- `dim borrow <file>` - Run borrow checker
- `dim build <file>` - Full pipeline
- `dim test` - Run test suite

## Testing

**File:** `dim_tests.py`

Uses a simple test framework with:
- `@test` decorator for test registration
- Tag-based filtering
- Assertion helpers

Run tests:
```bash
python dim_tests.py
python dim_tests.py --tag lexer
```

## Future Work

### Phase 3 (In Progress)
- Native binary emission via LLVM
- WASM target
- Link-time optimization

### Phase 4
- Async/await runtime
- Actor message passing implementation

### Phase 5
- Typed prompts with model adapters
- Structured output validation

### Phase 6
- Taint analysis
- Capability-based security
- Z3 integration for verification
