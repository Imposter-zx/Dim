# dim_cli.py — Dim Compiler CLI (v0.2)
#
# Unified command-line driver for the Dim compiler.
# Usage:  python dim_cli.py <command> [options] [file]

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
    lexer = Lexer(source, args.file)
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
    ast = parser.parse_program()
    _print_ast(ast, indent=0)
    parser.diag.flush(color=not args.no_color)


def cmd_check(args):
    """Type-check and semantically analyse a .dim file."""
    from dim_lexer import Lexer
    from dim_parser import Parser
    from dim_semantic import SemanticAnalyzer
    from dim_module_resolver import ModuleResolver

    source = _load_file(args.file)
    tokens = Lexer(source, args.file).tokenize()
    parser = Parser(tokens, source, args.file)
    ast = parser.parse_program()
    if parser.diag.has_errors:
        parser.diag.flush(color=not args.no_color)
        sys.exit(1)
    module_resolver = ModuleResolver(args.file)
    sem = SemanticAnalyzer(source, args.file, module_resolver)
    module_resolver.resolve_program(ast, source, args.file)
    ok = sem.analyze(ast)
    if sem.diag.has_errors:
        sem.diag.flush(color=not args.no_color)
        sys.exit(1)
    print("[PASS] Type checking passed.")


def cmd_mir(args):
    """Lower a .dim file to MIR and print the CFG."""
    from dim_lexer import Lexer
    from dim_parser import Parser
    from dim_semantic import SemanticAnalyzer
    from dim_mir_lowering import lower_program
    from dim_module_resolver import ModuleResolver

    source = _load_file(args.file)
    tokens = Lexer(source, args.file).tokenize()
    parser = Parser(tokens, source, args.file)
    ast = parser.parse_program()
    module_resolver = ModuleResolver(args.file)
    sem = SemanticAnalyzer(source, args.file, module_resolver)
    module_resolver.resolve_program(ast, source, args.file)
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
    from dim_module_resolver import ModuleResolver

    source = _load_file(args.file)
    tokens = Lexer(source, args.file).tokenize()
    parser = Parser(tokens, source, args.file)
    ast = parser.parse_program()
    module_resolver = ModuleResolver(args.file)
    sem = SemanticAnalyzer(source, args.file, module_resolver)
    module_resolver.resolve_program(ast, source, args.file)
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
        print("[PASS] Borrow check passed.")
    else:
        sys.exit(1)


def cmd_build(args):
    """Full pipeline: lex → parse → type-check → MIR → borrow-check → native binary."""
    print("[1/5] Lexing and Parsing...")
    from dim_lexer import Lexer
    from dim_parser import Parser
    from dim_semantic import SemanticAnalyzer
    from dim_mir_lowering import lower_program
    from dim_borrow_checker import BorrowChecker
    from dim_diagnostic import DiagnosticBag
    from dim_module_resolver import ModuleResolver

    source = _load_file(args.file)
    tokens = Lexer(source, args.file).tokenize()
    parser = Parser(tokens, source, args.file)
    ast = parser.parse_program()
    if parser.diag.has_errors:
        parser.diag.flush(color=not args.no_color)
        sys.exit(1)

    print("[2/5] Type checking...")
    module_resolver = ModuleResolver(args.file)
    sem = SemanticAnalyzer(source, args.file, module_resolver)
    module_resolver.resolve_program(ast, source, args.file)
    ok = sem.analyze(ast)
    sem.diag.flush(color=not args.no_color)
    if not ok:
        sys.exit(1)

    print("[3/5] Lowering to MIR...")
    module = lower_program(ast)

    print("[4/5] Borrow checking...")
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

    print("[5/5] Generating LLVM IR...")
    from dim_mir_to_llvm import LLVMGenerator

    gen = LLVMGenerator()
    llvm_ir = gen.generate(module)
    print(f"  LLVM IR generated ({len(llvm_ir)} bytes)")

    if args.output:
        output = args.output
    else:
        base = os.path.splitext(os.path.basename(args.file))[0]
        output = base + ".ll"

    with open(output, "w") as f:
        f.write(llvm_ir)
    print(f"  LLVM IR written to {output}")

    print("\n[PASS] Build succeeded for " + args.file)

    native = getattr(args, "native", False)
    if native:
        print("\nAttempting native binary generation...")
        try:
            from dim_native_codegen import emit_native

            binary = emit_native(llvm_ir, output.replace(".ll", ""))
            if binary:
                print(f"  Native binary: {binary}")
            else:
                print("  (Install LLVM tools for native binaries)")
        except Exception as e:
            print(f"  Native compilation skipped: {e}")


def cmd_test(args):
    """Run the built-in test suite."""
    import dim_tests

    tag = getattr(args, "tag", None)
    ok = dim_tests.run_tests(filter_tag=tag)
    sys.exit(0 if ok else 1)


def cmd_pkg(args):
    """Package manager commands."""
    from dim_pkg import run_pkg

    run_pkg(args.subargs)


def cmd_run(args):
    """Build and run a .dim file."""
    from dim_build import BuildSystem

    project_path = os.path.dirname(os.path.abspath(args.file)) or os.getcwd()
    builder = BuildSystem(project_path)
    builder.config.main = os.path.basename(args.file)
    success = builder.run(args.args)
    sys.exit(0 if success else 1)


def cmd_new(args):
    """Create a new Dim project."""
    from dim_build import BuildSystem

    project_path = os.path.join(os.getcwd(), args.name)
    if os.path.exists(project_path):
        print(f"error: directory '{args.name}' already exists")
        sys.exit(1)
    os.makedirs(project_path)
    builder = BuildSystem(project_path)
    builder.init(args.name, args.main or "main.dim")
    print(f"\nCreated project '{args.name}'")
    print(f"  cd {args.name}")
    print(f"  dim run {args.main or 'main.dim'}")


def cmd_bench(args):
    """Run benchmarks."""
    print("Benchmark functionality coming soon...")
    sys.exit(0)


def cmd_fmt(args):
    """Format a .dim file."""
    from dim_formatter import format_file

    output = getattr(args, "output", None)
    indent = getattr(args, "indent", 4)
    format_file(args.file, output, indent)


def _print_ast(node, indent: int = 0):
    from dim_ast import Node

    prefix = "  " * indent
    name = type(node).__name__
    fields = {
        k: v
        for k, v in vars(node).items()
        if k not in ("span", "resolved_type", "resolved_fn_type")
    }
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
    parser.add_argument(
        "--no-color", action="store_true", help="Disable ANSI color output"
    )
    sub = parser.add_subparsers(dest="command")

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
    p_build.add_argument("-o", "--output", default=None, help="Output file")
    p_build.add_argument(
        "--native", action="store_true", help="Generate native binary (requires LLVM)"
    )
    p_build.set_defaults(func=cmd_build)

    # test
    p_test = sub.add_parser("test", help="Run the compiler test suite")
    p_test.add_argument("--tag", default=None, help="Filter tests by tag")
    p_test.set_defaults(func=cmd_test)

    # pkg
    p_pkg = sub.add_parser("pkg", help="Package manager")
    p_pkg.add_argument("subargs", nargs="*", default=[], help="Package subcommand")
    p_pkg.set_defaults(func=cmd_pkg)

    # run
    p_run = sub.add_parser("run", help="Build and run a .dim file")
    p_run.add_argument("file", nargs="?", default=None, help=".dim file to run")
    p_run.add_argument("args", nargs="*", default=[], help="Arguments to pass")
    p_run.set_defaults(func=cmd_run)

    # new
    p_new = sub.add_parser("new", help="Create a new Dim project")
    p_new.add_argument("name", help="Project name")
    p_new.add_argument("-m", "--main", default=None, help="Main file name")
    p_new.set_defaults(func=cmd_new)

    # bench
    p_bench = sub.add_parser("bench", help="Run benchmarks")
    p_bench.add_argument("file", help="Benchmark file")
    p_bench.set_defaults(func=cmd_bench)

    # fmt
    p_fmt = sub.add_parser("fmt", help="Format a .dim file")
    p_fmt.add_argument("file")
    p_fmt.add_argument(
        "-o", "--output", default=None, help="Output file (default: stdout)"
    )
    p_fmt.add_argument(
        "-i", "--indent", type=int, default=4, help="Indent size (default: 4)"
    )
    p_fmt.set_defaults(func=cmd_fmt)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
