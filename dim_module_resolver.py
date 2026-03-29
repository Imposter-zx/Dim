# dim_module_resolver.py — Module Resolution for Dim
#
# Handles finding, parsing, and caching of imported modules.
# Supports relative and absolute module paths.

import os
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from dim_ast import Program, Statement, FunctionDef, StructDef, EnumDef, ImportStmt
from dim_lexer import Lexer
from dim_parser import Parser
from dim_semantic import SemanticAnalyzer
from dim_diagnostic import DiagnosticBag


@dataclass
class Module:
    name: str
    path: str
    program: Program
    exports: Dict[str, Statement] = field(default_factory=dict)
    symbols: Dict[str, object] = field(default_factory=dict)
    imported: bool = False


class ModuleResolver:
    """
    Resolves import statements to loaded modules.
    Maintains a cache of already-loaded modules to avoid duplicate parsing.
    """

    def __init__(self, entry_file: str):
        self.entry_file = os.path.abspath(entry_file)
        self.entry_dir = os.path.dirname(self.entry_file)
        self._cache: Dict[str, Module] = {}
        self._loading: Set[str] = set()
        self._stdlib_dir = self._find_stdlib_dir()

    def _find_stdlib_dir(self) -> Optional[str]:
        candidates = [
            os.path.join(os.path.dirname(self.entry_file), "std"),
            os.path.join(os.path.dirname(__file__), "std"),
            os.path.join(os.getcwd(), "std"),
        ]
        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                return os.path.normpath(candidate)
        return None

    def _get_module_path(self, path: List[str]) -> Optional[str]:
        if path[0] == "std":
            if self._stdlib_dir:
                return os.path.join(self._stdlib_dir, *path[1:]) + ".dim"
            return None

        module_path = os.path.join(self.entry_dir, *path) + ".dim"
        if os.path.isfile(module_path):
            return module_path

        module_path = os.path.join(self.entry_dir, *path, "__init__.dim")
        if os.path.isfile(module_path):
            return module_path

        return None

    def resolve(
        self, import_stmt: ImportStmt, source: str, filename: str
    ) -> Optional[Module]:
        path = import_stmt.path
        module_name = ".".join(path)

        if module_name in self._cache:
            return self._cache[module_name]

        if module_name in self._loading:
            return None

        file_path = self._get_module_path(path)
        if not file_path:
            return None

        self._loading.add(module_name)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                module_source = f.read()
        except Exception:
            self._loading.discard(module_name)
            return None

        lexer = Lexer(module_source, file_path)
        tokens = lexer.tokenize()
        if lexer.diag.has_errors:
            self._loading.discard(module_name)
            return None

        parser = Parser(tokens, module_source, file_path)
        program = parser.parse_program()
        if parser.diag.has_errors:
            self._loading.discard(module_name)
            return None

        sem = SemanticAnalyzer(module_source, file_path)
        sem.analyze(program)
        if sem.diag.has_errors:
            self._loading.discard(module_name)
            return None

        exports = self._collect_exports(program)

        module = Module(
            name=module_name,
            path=file_path,
            program=program,
            exports=exports,
            imported=True,
        )

        self._cache[module_name] = module
        self._loading.discard(module_name)
        return module

    def _collect_exports(self, program: Program) -> Dict[str, Statement]:
        exports = {}
        for stmt in program.statements:
            if isinstance(stmt, FunctionDef):
                exports[stmt.name] = stmt
            elif isinstance(stmt, StructDef):
                exports[stmt.name] = stmt
            elif isinstance(stmt, EnumDef):
                exports[stmt.name] = stmt
            elif isinstance(stmt, ImportStmt) and stmt.alias:
                exports[stmt.alias] = stmt
        return exports

    def resolve_program(
        self, program: Program, source: str, filename: str
    ) -> Dict[str, Module]:
        for stmt in program.statements:
            if isinstance(stmt, ImportStmt):
                self.resolve(stmt, source, filename)
        return self._cache
