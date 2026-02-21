# dim_semantic.py — Semantic Analyzer for Dim (v0.2)
#
# Updated to use the new type system (dim_types), structured Diagnostics,
# Token dataclass, and span-annotated AST.
# Orchestrates: name resolution → type checking → (future) MIR lowering

from __future__ import annotations
from typing import Dict, List, Optional

from dim_ast import *
from dim_types import Type, FunctionType, UNIT, UnknownType, resolve_builtin
from dim_type_checker import TypeChecker, TypeEnv, Symbol
from dim_diagnostic import DiagnosticBag


class SemanticAnalyzer:
    """
    Top-level semantic analysis pass.
    Wraps the TypeChecker and exposes a simple analyze() interface.
    After analysis, call .diag to inspect errors/warnings.
    """

    def __init__(self, source: str = "", filename: str = "<stdin>"):
        self.source   = source
        self.filename = filename
        self.tc       = TypeChecker(source, filename)
        self.diag     = self.tc.diag

    def analyze(self, program: Program) -> bool:
        """
        Perform full semantic analysis on a Program node.
        Returns True if there are no errors, False otherwise.
        """
        self.tc.check_program(program)
        return not self.diag.has_errors

    def report(self, color: bool = True) -> str:
        """Return all collected diagnostics as a formatted string."""
        import io, sys
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        self.diag.flush(color=color)
        sys.stderr = old
        return buf.getvalue()


# ── Example / smoke-test when run directly ────────────────────────────────────
if __name__ == "__main__":
    from dim_lexer import Lexer
    from dim_parser import Parser

    CODE = """\
fn add(x: i32, y: i32) -> i32:
    return x + y

fn main():
    let result = add(10, 20)
    let mut counter = 0
    counter = counter + 1
    let bad = unknown_var
"""
    print("=== Dim Semantic Analyzer v0.2 ===")
    print(f"Source:\n{CODE}\n")

    lexer  = Lexer(CODE, "test.dim")
    tokens = lexer.tokenize()

    parser = Parser(tokens, CODE, "test.dim")
    ast    = parser.parse_program()

    sem = SemanticAnalyzer(CODE, "test.dim")
    ok  = sem.analyze(ast)

    print("=== Diagnostics ===")
    sem.diag.flush(color=True)
    print(f"\n=== Analysis {'PASSED' if ok else 'FAILED with errors'} ===")
