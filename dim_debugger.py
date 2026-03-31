# dim_debugger.py — Interactive Debugger for Dim
#
# Provides breakpoint management, step-through debugging, variable inspection.

import sys
import os
import subprocess
import tempfile
from typing import Optional, Dict, Any, List, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from dim_lexer import Lexer
from dim_parser import Parser
from dim_semantic import SemanticAnalyzer
from dim_module_resolver import ModuleResolver
from dim_mir_lowering import lower_program, MIRFunction, MIRBasicBlock
from dim_mir_to_llvm import LLVMGenerator


class DebugCommand(Enum):
    CONTINUE = "continue"
    STEP = "step"
    NEXT = "next"
    FINISH = "finish"
    BREAK = "break"
    DELETE = "delete"
    LIST = "list"
    PRINT = "print"
    EVAL = "eval"
    BACKTRACE = "backtrace"
    QUIT = "quit"


@dataclass
class Breakpoint:
    id: int
    file: str
    line: int
    condition: Optional[str] = None
    enabled: bool = True
    hits: int = 0


@dataclass
class StackFrame:
    level: int
    function: str
    file: str
    line: int
    locals: Dict[str, Any]


class DimDebugger:
    def __init__(self, source_file: str):
        self.source_file = source_file
        self.breakpoints: Dict[int, Breakpoint] = {}
        self.next_bp_id = 1

        self.compiled = False
        self.executable: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None

        self.current_line: Optional[int] = None
        self.stack: List[StackFrame] = []

        self.source_lines: List[str] = []
        self._load_source()

    def _load_source(self):
        with open(self.source_file, "r") as f:
            self.source_lines = f.read().splitlines()

    def compile(self, output_file: Optional[str] = None) -> bool:
        if not output_file:
            output_file = tempfile.mktemp(suffix=".exe")

        with open(self.source_file, "r") as f:
            source = f.read()

        try:
            tokens = Lexer(source, self.source_file).tokenize()
            parser = Parser(tokens, source, self.source_file)
            ast = parser.parse_program()

            if parser.diag.has_errors:
                print("Compilation errors:")
                for err in parser.diag.errors:
                    print(f"  {err}")
                return False

            resolver = ModuleResolver(self.source_file)
            sem = SemanticAnalyzer(source, self.source_file, resolver)
            resolver.resolve_program(ast, source, self.source_file)

            if not sem.analyze(ast):
                print("Type errors:")
                for err in sem.diag.errors:
                    print(f"  {err}")
                return False

            module = lower_program(ast)
            gen = LLVMGenerator(platform="windows")
            llvm_ir = gen.generate(module)

            runtime_path = os.path.join(
                os.path.dirname(__file__), "runtime", "dim_runtime.c"
            )

            with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
                f.write(llvm_ir)
                ll_path = f.name

            result = subprocess.run(
                ["clang", "-O0", "-g", "-o", output_file, ll_path, runtime_path],
                capture_output=True,
                text=True,
            )

            os.unlink(ll_path)

            if result.returncode != 0:
                print("Compilation failed:")
                print(result.stderr)
                return False

            self.executable = output_file
            self.compiled = True
            return True

        except Exception as e:
            print(f"Compilation error: {e}")
            return False

    def add_breakpoint(self, line: int, condition: Optional[str] = None) -> Breakpoint:
        bp = Breakpoint(
            id=self.next_bp_id, file=self.source_file, line=line, condition=condition
        )
        self.breakpoints[self.next_bp_id] = bp
        self.next_bp_id += 1
        print(f"Breakpoint {bp.id} set at line {line}")
        return bp

    def delete_breakpoint(self, bp_id: int) -> bool:
        if bp_id in self.breakpoints:
            del self.breakpoints[bp_id]
            print(f"Breakpoint {bp_id} deleted")
            return True
        print(f"Breakpoint {bp_id} not found")
        return False

    def list_breakpoints(self):
        if not self.breakpoints:
            print("No breakpoints set")
            return

        print("Breakpoints:")
        for bp in self.breakpoints.values():
            status = "enabled" if bp.enabled else "disabled"
            cond = f" if {bp.condition}" if bp.condition else ""
            print(f"  {bp.id}: {bp.file}:{bp.line} ({status}){cond}")

    def list_source(self, start: int, end: int):
        for i in range(start, min(end + 1, len(self.source_lines) + 1)):
            marker = ">" if self.current_line == i else " "
            bp_markers = []
            for bp in self.breakpoints.values():
                if bp.line == i and bp.enabled:
                    bp_markers.append(f"B{bp.id}")
            bp_str = ", ".join(bp_markers) if bp_markers else ""
            print(f"{i:4} {marker} {bp_str:8} {self.source_lines[i - 1]}")

    def run(self, args: Optional[List[str]] = None):
        if not self.compiled:
            if not self.compile():
                return

        print(f"Running {self.source_file}...")

        cmd = [self.executable] + (args or [])
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        stdout, stderr = self.process.communicate()

        print("Program output:")
        if stdout:
            print(stdout)
        if stderr:
            print("Errors:", stderr)

        print(f"Program exited with code {self.process.returncode}")

    def start_debug(self, args: Optional[List[str]] = None):
        if not self.compiled:
            if not self.compile():
                return

        print(f"Starting debug session for {self.source_file}")
        print(
            "Commands: continue, step, next, finish, break, delete, list, print, eval, quit"
        )
        print()

        print("Note: Debugger currently runs program to completion.")
        print("      Breakpoints are reported but execution continues.")

        cmd = [self.executable] + (args or [])

        if self.breakpoints:
            print(f"\nBreakpoints set:")
            for bp in self.breakpoints.values():
                print(f"  Line {bp.line}")

        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        while True:
            line = input("(dbg) ").strip()

            if not line:
                continue

            parts = line.split()
            cmd = parts[0]

            if cmd == "quit" or cmd == "q":
                if self.process:
                    self.process.terminate()
                print("Exiting debugger...")
                break
            elif cmd == "continue" or cmd == "c":
                print("Continuing...")
                break
            elif cmd == "list" or cmd == "l":
                start = 1
                end = min(20, len(self.source_lines))
                if len(parts) > 1:
                    try:
                        start = int(parts[1])
                        end = start + 19
                    except ValueError:
                        pass
                self.list_source(start, end)
            elif cmd == "break" or cmd == "b":
                if len(parts) > 1:
                    try:
                        line = int(parts[1])
                        self.add_breakpoint(line)
                    except ValueError:
                        print("Usage: break <line>")
                else:
                    print("Usage: break <line>")
            elif cmd == "delete" or cmd == "d":
                if len(parts) > 1:
                    try:
                        bp_id = int(parts[1])
                        self.delete_breakpoint(bp_id)
                    except ValueError:
                        print("Usage: delete <id>")
                else:
                    print("Usage: delete <id>")
            elif cmd == "print" or cmd == "p":
                print("Variable inspection not available in current mode")
            elif cmd in ("help", "?"):
                print("Commands:")
                print("  continue, c   - Continue execution")
                print("  list, l      - List source lines")
                print("  break, b     - Set breakpoint")
                print("  delete, d   - Delete breakpoint")
                print("  print, p    - Print variable")
                print("  quit, q     - Exit debugger")
            else:
                print(f"Unknown command: {cmd}")

        stdout, stderr = self.process.communicate()

        print("\nProgram output:")
        if stdout:
            print(stdout)
        if stderr:
            print("Errors:", stderr)


def run_debugger(source_file: str, args: Optional[List[str]] = None):
    debugger = DimDebugger(source_file)
    debugger.start_debug(args)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dim_debugger.py <source.dim> [args...]")
        sys.exit(1)

    source_file = sys.argv[1]
    args = sys.argv[2:] if len(sys.argv) > 2 else None

    run_debugger(source_file, args)
