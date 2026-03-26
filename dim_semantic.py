# dim_semantic.py — Semantic Analyzer for Dim (v0.2)
#
# Updated to use the new type system (dim_types), structured Diagnostics,
# Token dataclass, and span-annotated AST.
# Orchestrates: name resolution → type checking → (future) MIR lowering

from typing import Dict, List, Optional, Set

from dim_ast import *
from dim_types import Type, FunctionType, UNIT, UnknownType, resolve_builtin
from dim_type_checker import TypeChecker, TypeEnv, Symbol
from dim_diagnostic import DiagnosticBag


CAPABILITY_REGISTRY: Dict[str, Set[str]] = {}


class SemanticAnalyzer:
    """
    Top-level semantic analysis pass.
    Wraps the TypeChecker and exposes a simple analyze() interface.
    After analysis, call .diag to inspect errors/warnings.
    """

    KNOWN_CAPABILITIES: Set[str] = {
        "NetRead",
        "NetWrite",
        "FileRead",
        "FileWrite",
        "FileExecute",
        "Crypto",
        "System",
        "Process",
        "HttpGet",
        "HttpPost",
        "SqlQuery",
        "EnvRead",
    }

    def __init__(self, source: str = "", filename: str = "<stdin>"):
        self.source = source
        self.filename = filename
        self.tc = TypeChecker(source, filename)
        self.diag = self.tc.diag
        self._tools: Dict[str, ToolDecorator] = {}
        self._capability_calls: List[tuple] = []

    def analyze(self, program: Program) -> bool:
        self._register_tools(program)
        self._validate_capabilities(program)
        self.tc.check_program(program)
        return not self.diag.has_errors

    def _register_tools(self, program: Program):
        for s in program.statements:
            if isinstance(s, FunctionDef) and s.tool:
                tool = s.tool
                self._tools[s.name] = tool
                for perm in tool.permissions:
                    cap = perm.split("(")[0]
                    if cap not in self.KNOWN_CAPABILITIES:
                        self.diag.warning(
                            "W0100",
                            f"Unknown capability `{cap}` in @tool `{tool.name}`",
                            tool.span,
                        )

    def _validate_capabilities(self, program: Program):
        for s in program.statements:
            self._check_stmt_caps(s)

    def _check_stmt_caps(self, stmt: Statement):
        if isinstance(stmt, FunctionDef):
            for s in stmt.body:
                self._check_stmt_caps(s)
            if stmt.tool:
                for perm in stmt.tool.permissions:
                    cap = perm.split("(")[0]
                    if cap not in self.KNOWN_CAPABILITIES:
                        self.diag.error(
                            "E0061",
                            f"Unknown capability `{cap}` in @tool decorator",
                            stmt.tool.span,
                            hints=[
                                "Available: "
                                + ", ".join(sorted(self.KNOWN_CAPABILITIES))
                            ],
                        )
        elif isinstance(stmt, IfStmt):
            for s in stmt.then_branch:
                self._check_stmt_caps(s)
            for _, body in stmt.elif_branches:
                for s in body:
                    self._check_stmt_caps(s)
            if stmt.else_branch:
                for s in stmt.else_branch:
                    self._check_stmt_caps(s)
        elif isinstance(stmt, WhileStmt):
            for s in stmt.body:
                self._check_stmt_caps(s)
        elif isinstance(stmt, ForStmt):
            for s in stmt.body:
                self._check_stmt_caps(s)
        elif isinstance(stmt, MatchStmt):
            for arm in stmt.arms:
                for s in arm.body:
                    self._check_stmt_caps(s)

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

    lexer = Lexer(CODE, "test.dim")
    tokens = lexer.tokenize()

    parser = Parser(tokens, CODE, "test.dim")
    ast = parser.parse_program()

    sem = SemanticAnalyzer(CODE, "test.dim")
    ok = sem.analyze(ast)

    print("=== Diagnostics ===")
    sem.diag.flush(color=True)
    print(f"\n=== Analysis {'PASSED' if ok else 'FAILED with errors'} ===")
