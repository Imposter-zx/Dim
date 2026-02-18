# Dim Semantic Analyzer (POC)

from dim_ast import *

class SemanticAnalyzer:
    def __init__(self):
        self.scopes = [{}] # Stack of symbol tables

    def enter_scope(self):
        self.scopes.append({})

    def exit_scope(self):
        self.scopes.pop()

    def define(self, name, type_info):
        self.scopes[-1][name] = type_info

    def lookup(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def analyze(self, node):
        if isinstance(node, Program):
            for stmt in node.statements:
                self.analyze(stmt)
        elif isinstance(node, FunctionDef):
            self.define(node.name, {"type": "function", "return": node.return_type})
            self.enter_scope()
            # Define parameters in the function scope
            for param_name, param_type in node.params:
                self.define(param_name, {"type": param_type})
            for stmt in node.body:
                self.analyze(stmt)
            self.exit_scope()
        elif isinstance(node, IfStmt):
            self.analyze(node.condition)
            self.enter_scope()
            for stmt in node.then_branch:
                self.analyze(stmt)
            self.exit_scope()
            if node.else_branch:
                self.enter_scope()
                for stmt in node.else_branch:
                    self.analyze(stmt)
                self.exit_scope()
        elif isinstance(node, WhileStmt):
            self.analyze(node.condition)
            self.enter_scope()
            for stmt in node.body:
                self.analyze(stmt)
            self.exit_scope()
        elif isinstance(node, ForStmt):
            self.enter_scope()
            self.define(node.iterator, {"type": "unknown"}) # Iterator type depends on iterable
            self.analyze(node.iterable)
            for stmt in node.body:
                self.analyze(stmt)
            self.exit_scope()
        elif isinstance(node, PromptDef):
            self.define(node.name, {"type": "prompt"})
        elif isinstance(node, LetStmt):
            res = self.analyze(node.value)
            self.define(node.name, {"type": res or "unknown"})
        elif isinstance(node, ReturnStmt):
            if node.value:
                return self.analyze(node.value)
        elif isinstance(node, Literal):
            if isinstance(node.value, int): return "int"
            if isinstance(node.value, float): return "float"
            if isinstance(node.value, str): return "string"
            if isinstance(node.value, bool): return "bool"
        elif isinstance(node, Identifier):
            info = self.lookup(node.name)
            if info is None:
                print(f"Error: Undefined variable '{node.name}'")
            return info.get("type") if info else None
        elif isinstance(node, BinaryOp):
            lt = self.analyze(node.left)
            rt = self.analyze(node.right)
            if lt and rt and lt != rt:
                print(f"Warning: Type mismatch in '{node.op}': {lt} and {rt}")
            return lt
        elif isinstance(node, Call):
            self.analyze(node.callee)
            for arg in node.args:
                self.analyze(arg)
            return "unknown" # Function return type lookup not fully implemented

# POC Execution Integration
if __name__ == "__main__":
    from dim_poc_lexer import Lexer
    from dim_parser import Parser
    code = """
fn main():
    let x = 42
    let y = "hello"
    let z = x + y
    if x > 10:
        let internal = 5
    let w = internal
"""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse_program()
    analyzer = SemanticAnalyzer()
    print("--- Starting Semantic Analysis ---")
    analyzer.analyze(ast)
    print("--- Semantic Analysis Complete ---")
