# dim_repl.py — Read-Eval-Print Loop for Dim
#
# Interactive interpreter for the Dim language using tree-walking interpreter.

import sys
import os
from typing import Optional, Dict, Any, List
from dim_interpreter import DimInterpreter, Environment, RuntimeValue


class DimREPL:
    def __init__(self):
        self.interpreter = DimInterpreter()
        self.env = Environment(self.interpreter.global_env)
        self.buffer: List[str] = []
        self.in_function = False
        self.function_name = ""
        self.function_params: List[str] = []

    def run(self):
        print("Dim REPL v0.6.0 (Interpreter Mode)")
        print("Type :help for commands, :quit to exit")
        print("Type multi-line code with proper indentation")
        print()

        while True:
            try:
                if self.in_function:
                    prompt = "fn> "
                else:
                    prompt = "dim> "
                line = input(prompt)
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not line.strip() and not self.in_function:
                continue

            if line.strip().startswith(":"):
                self._handle_command(line)
                continue

            if self.in_function:
                self._handle_multiline_input(line)
            else:
                self._handle_input(line)

    def _handle_input(self, line: str):
        if not self.buffer and self._starts_function(line):
            self._start_function_block(line)
            return

        if self.buffer:
            self._handle_multiline_input(line)
            return

        result = self._eval_line(line)
        if result is not None:
            print(result)

    def _starts_function(self, line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("fn ") and ":" in stripped

    def _start_function_block(self, line: str):
        parts = line.split(":")[0].strip().split()
        if len(parts) >= 2:
            self.function_name = parts[1]
            self.function_params = []
            if "(" in line:
                params_str = line.split("(")[1].split(")")[0]
                if params_str.strip():
                    self.function_params = [p.strip() for p in params_str.split(",")]

            self.in_function = True
            self.buffer = []
            self.buffer.append(line)
            print(f"  (Defining function {self.function_name}...)")

    def _handle_multiline_input(self, line: str):
        if line.strip() == "":
            if self.in_function:
                self.buffer.append("")
                return

        if (
            self.in_function
            and line
            and not line[0].isspace()
            and not line.strip().startswith("fn")
        ):
            if line.strip() == "" or "return" in self.buffer[-1]:
                self._finish_function()
                return

        self.buffer.append(line)

    def _finish_function(self):
        code = "\n".join(self.buffer)

        from dim_lexer import Lexer
        from dim_parser import Parser

        try:
            tokens = Lexer(code, "<repl>").tokenize()
            parser = Parser(tokens, code, "<repl>")
            ast = parser.parse_program()

            for stmt in ast.statements:
                self.interpreter._execute_stmt(stmt, self.env)

            print(f"  ✓ Function '{self.function_name}' defined")

        except Exception as e:
            print(f"Error defining function: {e}")

        self.buffer = []
        self.in_function = False
        self.function_name = ""
        self.function_params = []

    def _eval_line(self, line: str) -> Optional[str]:
        from dim_lexer import Lexer
        from dim_parser import Parser

        try:
            code = f"fn __repl_main():\n    {line}\n    return none"

            tokens = Lexer(code, "<repl>").tokenize()
            parser = Parser(tokens, code, "<repl>")
            ast = parser.parse_program()

            for stmt in ast.statements:
                if hasattr(stmt, "body"):
                    for s in stmt.body:
                        self.interpreter._execute_stmt(s, self.env)

            main_func = self.env.get("__repl_main")
            if main_func and hasattr(main_func, "body"):
                main_env = Environment(self.env)
                result = None
                for stmt in main_func.body:
                    result = self.interpreter._execute_stmt(stmt, main_env)

                if result is not None:
                    return self.interpreter._to_string(result)

            return None

        except Exception as e:
            if "Undefined variable" in str(e):
                print(f"Error: {e}")
            else:
                print(f"Error: {e}")
            return None

    def _handle_command(self, cmd: str):
        parts = cmd.split()
        name = parts[0]

        if name == ":help":
            print("Commands:")
            print("  :quit, :exit  - Exit the REPL")
            print("  :clear       - Clear the screen")
            print("  :help        - Show this help")
            print("  :env         - Show current environment")
            print("  :type <expr> - Show type of expression")
            print("  :funcs       - List defined functions")
            print()
            print("Examples:")
            print("  5 + 3              => 8")
            print("  x = 10             => defines variable")
            print("  fn add(a, b):      => starts function definition")
            print("    return a + b")
            print("                     => (empty line ends function)")

        elif name in (":quit", ":exit"):
            sys.exit(0)

        elif name == ":clear":
            os.system("cls" if os.name == "nt" else "clear")

        elif name == ":env":
            print("Variables:")
            for name, value in self.env.variables.items():
                print(f"  {name}: {self.interpreter._to_string(value)}")

            funcs = [n for n, v in self.env.functions.items() if n != "__repl_main"]
            if funcs:
                print("Functions:")
                for fn in funcs:
                    print(f"  {fn}")

        elif name == ":type":
            if len(parts) > 1:
                expr = " ".join(parts[1:])
                print(f"Type of '{expr}': (showing value)")
                result = self._eval_line(expr)
                if result:
                    print(f"  => {result}")
            else:
                print("Usage: :type <expression>")

        elif name == ":funcs":
            funcs = [n for n, v in self.env.functions.items() if n != "__repl_main"]
            if funcs:
                print("Defined functions:")
                for fn in funcs:
                    print(f"  {fn}")
            else:
                print("No functions defined")

        else:
            print(f"Unknown command: {name}")


def run_repl():
    repl = DimREPL()
    repl.run()


if __name__ == "__main__":
    run_repl()
