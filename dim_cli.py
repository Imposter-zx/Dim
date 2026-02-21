# dim_cli.py — Dim Compiler CLI (v0.2)
#
# Unified command-line driver for the Dim compiler.
# Usage:  python dim_cli.py <command> [options] [file]

from __future__ import annotations
import sys
import os
import argparse


def _load_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
        sys.exit(1)


def cmd_lex(args):
    """Tokenise a .dim file and print all tokens."""
    from dim_lexer import Lexer
    source = _load_file(args.file)
    lexer  = Lexer(source, args.file)
    tokens = lexer.tokenize()
    for tok in tokens:
        print(tok)
    lexer.diag.flush(color=not args.no_color)


def cmd_parse(args):
    """Parse a .dim file and print the AST."""
    from dim_lexer import Lexer
    from dim_parser import Parser
    source = _load_file(args.file)
    tokens = Lexer(source, args.file).tokenize()
    parser = Parser(tokens, source, args.file)
    ast    = parser.parse_program()
    _print_ast(ast, indent=0)
    parser.diag.flush(color=not args.no_color)


def cmd_check(args):
    """Type-check and semantically analyse a .dim file."""
    from dim_lexer import Lexer
    from dim_parser import Parser
    from dim_semantic import SemanticAnalyzer
    source = _load_file(args.file)
    tokens = Lexer(source, args.file).tokenize()
    parser = Parser(tokens, source, args.file)
    ast    = parser.parse_program()
    if parser.diag.has_errors:
        parser.diag.flush(color=not args.no_color)
        sys.exit(1)
    sem = SemanticAnalyzer(source, args.file)
    ok  = sem.analyze(ast)
    sem.diag.flush(color=not args.no_color)
    if not ok:
        sys.exit(1)
    print("✓ Type checking passed.")


def cmd_mir(args):
    """Lower a .dim file to MIR and print the CFG."""
    from dim_lexer import Lexer
    from dim_parser import Parser
    from dim_semantic import SemanticAnalyzer
    from dim_mir_lowering import lower_program
    source = _load_file(args.file)
    tokens = Lexer(source, args.file).tokenize()
    parser = Parser(tokens, source, args.file)
    ast    = parser.parse_program()
    sem    = SemanticAnalyzer(source, args.file)
    sem.analyze(ast)
    sem.diag.flush(color=not args.no_color)
    module = lower_program(ast)
    for fn in module.functions:
        print(fn.pretty())


def cmd_borrow(args):
    """Run borrow checker on the MIR of a .dim file."""
    from dim_lexer import Lexer
    from dim_parser import Parser
    from dim_semantic import SemanticAnalyzer
    from dim_mir_lowering import lower_program
    from dim_borrow_checker import BorrowChecker
    from dim_diagnostic import DiagnosticBag
    source = _load_file(args.file)
    tokens = Lexer(source, args.file).tokenize()
    parser = Parser(tokens, source, args.file)
    ast    = parser.parse_program()
    sem    = SemanticAnalyzer(source, args.file)
    sem.analyze(ast)
    if sem.diag.has_errors:
        sem.diag.flush(color=not args.no_color)
        sys.exit(1)
    module = lower_program(ast)
    all_ok = True
    for fn in module.functions:
        diag = DiagnosticBag(source, args.file)
        checker = BorrowChecker(fn, diag)
        checker.check()
        diag.flush(color=not args.no_color)
        if diag.has_errors:
            all_ok = False
    if all_ok:
        print("✓ Borrow check passed.")
    else:
        sys.exit(1)


def cmd_build(args):
    """Full pipeline: lex → parse → type-check → MIR → borrow-check → (future: codegen)."""
    print("[1/4] Lexing and Parsing...")
    from dim_lexer import Lexer
    from dim_parser import Parser
    from dim_semantic import SemanticAnalyzer
    from dim_mir_lowering import lower_program
    from dim_borrow_checker import BorrowChecker
    from dim_diagnostic import DiagnosticBag

    source = _load_file(args.file)
    tokens = Lexer(source, args.file).tokenize()
    parser = Parser(tokens, source, args.file)
    ast    = parser.parse_program()
    if parser.diag.has_errors:
        parser.diag.flush(color=not args.no_color)
        sys.exit(1)

    print("[2/4] Type checking...")
    sem = SemanticAnalyzer(source, args.file)
    ok  = sem.analyze(ast)
    sem.diag.flush(color=not args.no_color)
    if not ok:
        sys.exit(1)

    print("[3/4] Lowering to MIR...")
    module = lower_program(ast)

    print("[4/4] Borrow checking...")
    all_ok = True
    for fn in module.functions:
        diag = DiagnosticBag(source, args.file)
        checker = BorrowChecker(fn, diag)
        checker.check()
        diag.flush(color=not args.no_color)
        if diag.has_errors:
            all_ok = False
    if not all_ok:
        sys.exit(1)

    print(f"\n✓ Build succeeded for {args.file}")
    print("  (Code generation / native binary output: coming in Phase 3)")


def cmd_test(args):
    """Run the built-in test suite."""
    import dim_tests
    tag = getattr(args, "tag", None)
    ok  = dim_tests.run_tests(filter_tag=tag)
    sys.exit(0 if ok else 1)


def _print_ast(node, indent: int = 0):
    from dim_ast import Node
    prefix = "  " * indent
    name   = type(node).__name__
    fields = {k: v for k, v in vars(node).items()
              if k not in ("span", "resolved_type", "resolved_fn_type")}
    print(f"{prefix}{name}")
    for key, val in fields.items():
        if isinstance(val, Node):
            print(f"{prefix}  {key}:")
            _print_ast(val, indent + 2)
        elif isinstance(val, list):
            if val and isinstance(val[0], Node):
                print(f"{prefix}  {key}: [")
                for item in val:
                    _print_ast(item, indent + 2)
                print(f"{prefix}  ]")
            elif val:
                print(f"{prefix}  {key}: {val}")
        elif val is not None:
            print(f"{prefix}  {key}: {val!r}")


def main():
    parser = argparse.ArgumentParser(
        prog="dim",
        description="Dim Programming Language Compiler (v0.2)",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color output")
    sub = parser.add_subparsers(dest="command", required=True)

    # lex
    p_lex = sub.add_parser("lex", help="Tokenise a .dim file")
    p_lex.add_argument("file")
    p_lex.set_defaults(func=cmd_lex)

    # parse
    p_parse = sub.add_parser("parse", help="Parse and print AST")
    p_parse.add_argument("file")
    p_parse.set_defaults(func=cmd_parse)

    # check
    p_check = sub.add_parser("check", help="Type-check a .dim file")
    p_check.add_argument("file")
    p_check.set_defaults(func=cmd_check)

    # mir
    p_mir = sub.add_parser("mir", help="Lower to MIR and print CFG")
    p_mir.add_argument("file")
    p_mir.set_defaults(func=cmd_mir)

    # borrow
    p_borrow = sub.add_parser("borrow", help="Run borrow checker")
    p_borrow.add_argument("file")
    p_borrow.set_defaults(func=cmd_borrow)

    # build
    p_build = sub.add_parser("build", help="Run full compiler pipeline")
    p_build.add_argument("file")
    p_build.set_defaults(func=cmd_build)

    # test
    p_test = sub.add_parser("test", help="Run the compiler test suite")
    p_test.add_argument("--tag", default=None, help="Filter tests by tag")
    p_test.set_defaults(func=cmd_test)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
