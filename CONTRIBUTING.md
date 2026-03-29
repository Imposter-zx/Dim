# Contributing to Dim

Thank you for your interest in contributing to the Dim programming language!

## Development Setup

### Prerequisites

- Python 3.6.8 or later
- Git

### Clone and Setup

```bash
git clone https://github.com/Imposter-zx/Dim.git
cd Dim
```

### Running Tests

```bash
# Run all tests
python dim_tests.py

# Run tests by tag
python dim_tests.py --tag lexer
python dim_tests.py --tag parser
python dim_tests.py --tag typecheck
python dim_tests.py --tag mir
python dim_tests.py --tag borrow
```

### Building

```bash
# Full pipeline build
python dim_cli.py build test.dim

# Individual stages
python dim_cli.py lex test.dim
python dim_cli.py parse test.dim
python dim_cli.py check test.dim
python dim_cli.py mir test.dim
python dim_cli.py borrow test.dim
```

## Project Structure

```
dim_token.py           — Token and Span definitions
dim_lexer.py           — Lexer (INDENT/DEDENT, keywords)
dim_parser.py          — Recursive descent parser
dim_ast.py             — AST node definitions
dim_types.py           — Type system
dim_type_checker.py    — Hindley-Milner type inference
dim_semantic.py        — Semantic analysis
dim_mir.py             — MIR (Mid-Level IR) structures
dim_mir_lowering.py    — AST → MIR lowering
dim_mir_to_llvm.py     — MIR → LLVM IR codegen
dim_borrow_checker.py  — Ownership & borrow checking
dim_diagnostic.py      — Error/warning system
dim_cli.py             — CLI interface
dim_tests.py           — Test suite
test.dim               — Example source file
```

## Adding Tests

Tests are in `dim_tests.py`. Use the `@test` decorator:

```python
@test("Description", "tag")
def test_name():
    # Test code
    pass
```

Available tags:
- `lexer` - Lexer tests
- `parser` - Parser tests
- `typecheck` - Type checker tests
- `mir` - MIR lowering tests
- `borrow` - Borrow checker tests
- `types` - Type system tests

## Coding Style

- Use meaningful variable names
- Add docstrings for public functions
- Keep functions focused and small
- Use type hints where appropriate

## Adding New Features

1. **Lexer**: Add tokens in `dim_token.py`, implement in `dim_lexer.py`
2. **Parser**: Add grammar rules in `dim_parser.py`
3. **AST**: Define nodes in `dim_ast.py`
4. **Type Checker**: Add rules in `dim_type_checker.py`
5. **MIR Lowering**: Add lowering in `dim_mir_lowering.py`
6. **LLVM Codegen**: Add codegen in `dim_mir_to_llvm.py`
7. **Tests**: Add test cases in `dim_tests.py`

## Reporting Issues

- Use GitHub Issues
- Include minimal reproduction steps
- Attach relevant error output
- Specify Python version

## Commit Messages

- Use clear, descriptive commit messages
- Start with type: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure all tests pass
5. Submit a pull request

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the work, not the person
