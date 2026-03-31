# dim_repl.py — Read-Eval-Print Loop for Dim
#
# Interactive interpreter for the Dim language.

import sys
import os
import readline
from typing import Optional, Dict, Any, List

from dim_lexer import Lexer
from dim_parser import Parser
from dim_semantic import SemanticAnalyzer
from dim_module_resolver import ModuleResolver
from dim_mir_lowering import lower_program
from dim_mir_to_llvm import LLVMGenerator
from dim_diagnostic import Diagnostic, DiagnosticKind


class DimREPL:
    def __init__(self):
        self.history: List[str] = []
        self.variables: Dict[str, Any] = {}
        self.parser: Optional[Parser] = None
        self.sem: Optional[SemanticAnalyzer] = None
        self.resolver: Optional[ModuleResolver] = None
        self.diag = Diagnostic()

    def load_stdlib(self):
        std_path = os.path.join(os.path.dirname(__file__), "std")
        for fname in os.listdir(std_path):
            if fname.endswith(".dim"):
                fpath = os.path.join(std_path, fname)
                with open(fpath, "r") as f:
                    source = f.read()
                self.eval_module(source, f"std/{fname}")

    def eval_module(self, source: str, filename: str = "<input>") -> bool:
        try:
            tokens = Lexer(source, filename).tokenize()
            parser = Parser(tokens, source, filename)
            ast = parser.parse_program()

            if parser.diag.has_errors:
                for err in parser.diag.errors:
                    print(f"Error: {err}")
                return False

            resolver = ModuleResolver(filename)
            sem = SemanticAnalyzer(source, filename, resolver)
            resolver.resolve_program(ast, source, filename)

            if not sem.analyze(ast):
                for err in sem.diag.errors:
                    print(f"Error: {err}")
                return False

            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def eval_expression(self, expr: str) -> Optional[Any]:
        source = f"let __repl_result = {expr}"

        try:
            tokens = Lexer(source, "<repl>").tokenize()
            parser = Parser(tokens, source, "<repl>")
            ast = parser.parse_program()

            if parser.diag.has_errors:
                for err in parser.diag.errors:
                    print(f"Error: {err}")
                return None

            resolver = ModuleResolver("<repl>")
            sem = SemanticAnalyzer(source, "<repl>", resolver)
            resolver.resolve_program(ast, source, "<repl>")

            if not sem.analyze(ast):
                for err in sem.diag.errors:
                    print(f"Error: {err}")
                return None

            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    def run(self):
        print("Dim REPL v0.5.0")
        print("Type :help for commands, :quit to exit")
        print()

        self.load_stdlib()

        while True:
            try:
                line = input("dim> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not line:
                continue

            if line.startswith(":"):
                self._handle_command(line)
                continue

            result = self.eval_expression(line)
            if result is not None:
                print(result)

    def _handle_command(self, cmd: str):
        parts = cmd.split()
        name = parts[0]

        if name == ":help":
            print("Commands:")
            print("  :quit, :exit  - Exit the REPL")
            print("  :clear       - Clear the screen")
            print("  :help        - Show this help")
            print("  :ast         - Show last parsed AST")
            print("  :env         - Show current environment")
        elif name in (":quit", ":exit"):
            sys.exit(0)
        elif name == ":clear":
            os.system("cls" if os.name == "nt" else "clear")
        elif name == ":env":
            print("Variables:", self.variables)
        elif name == ":ast":
            print("No AST available")
        else:
            print(f"Unknown command: {name}")


def run_repl():
    repl = DimREPL()
    repl.run()


if __name__ == "__main__":
    run_repl()
