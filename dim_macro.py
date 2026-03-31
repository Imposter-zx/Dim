# dim_macro.py — Macro System for Dim
#
# Provides compile-time macro expansion and code generation.

from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass

from dim_token import Token, TokenType, Span
from dim_ast import Node, Statement, Expression, Identifier, Literal


@dataclass
class MacroArg:
    name: str
    pattern: Optional[str] = None


@dataclass
class MacroRule:
    pattern: str
    template: str
    guard: Optional[str] = None


@dataclass
class MacroDefinition:
    name: str
    args: List[MacroArg]
    rules: List[MacroRule]
    body: Optional[str] = None
    span: Optional[Span] = None


class MacroExpander:
    def __init__(self):
        self.macros: Dict[str, MacroDefinition] = {}
        self.in_macro_expansion = False

    def define(self, macro: MacroDefinition):
        self.macros[macro.name] = macro

    def expand(self, node: Node) -> Node:
        return node

    def expand_program(self, stmts: List[Statement]) -> List[Statement]:
        expanded = []
        for stmt in stmts:
            expanded_stmt = self._expand_stmt(stmt)
            if expanded_stmt:
                expanded.append(expanded_stmt)
        return expanded

    def _expand_stmt(self, stmt: Statement) -> Optional[Statement]:
        return stmt

    def is_macro_call(self, name: str) -> bool:
        return name in self.macros


class MacroParser:
    @staticmethod
    def parse_definition(tokens: List[Token], start: int) -> Optional[MacroDefinition]:
        if start >= len(tokens) or tokens[start].type != TokenType.IDENT:
            return None

        name = tokens[start].value
        start += 1

        args = []
        if start < len(tokens) and tokens[start].type == TokenType.LPAREN:
            start += 1
            while start < len(tokens) and tokens[start].type != TokenType.RPAREN:
                if tokens[start].type == TokenType.IDENT:
                    arg_name = tokens[start].value
                    args.append(MacroArg(arg_name))
                    start += 1
                    if start < len(tokens) and tokens[start].type == TokenType.COMMA:
                        start += 1
            if start < len(tokens) and tokens[start].type == TokenType.RPAREN:
                start += 1

        rules = []
        body = None

        if start < len(tokens) and tokens[start].type == TokenType.LBRACE:
            start += 1
            body_start = start
            while start < len(tokens) and tokens[start].type != TokenType.RBRACE:
                start += 1
            body = "".join(t.value for t in tokens[body_start:start])

        return MacroDefinition(name, args, rules, body)

    @staticmethod
    def expand_template(template: str, args: Dict[str, str]) -> str:
        result = template
        for key, value in args.items():
            result = result.replace(f"${key}", value)
            result = result.replace(f"${{{key}}}", value)

        result = result.replace("$", "")

        return result


class MacroAttribute:
    @staticmethod
    def create_macro_call(name: str, args: List[Expression]) -> Expression:
        return Call(callee=Identifier(name), args=args, span=Span(0, 0, ""))


def macro_rules(rules: List[tuple]) -> Callable:
    def decorator(func: Callable) -> Callable:
        func._macro_rules = rules
        return func

    return decorator


BUILTIN_MACROS = {
    "debug": MacroDefinition(
        name="debug",
        args=[MacroArg("expr")],
        rules=[MacroRule("$expr", 'println!("DEBUG: $expr = ", $expr)')],
    ),
    "if_debug": MacroDefinition(
        name="if_debug", args=[MacroArg("block")], rules=[MacroRule("$block", "$block")]
    ),
    "todo": MacroDefinition(
        name="todo", args=[], rules=[MacroRule("", 'panic("TODO: not implemented")')]
    ),
    "unimplemented": MacroDefinition(
        name="unimplemented", args=[], rules=[MacroRule("", 'panic("unimplemented")')]
    ),
}


def register_builtin_macros(expander: MacroExpander):
    for name, macro in BUILTIN_MACROS.items():
        expander.define(macro)


if __name__ == "__main__":
    expander = MacroExpander()
    register_builtin_macros(expander)

    print("Available builtin macros:")
    for name, macro in BUILTIN_MACROS.items():
        print(f"  {name}({', '.join(a.name for a in macro.args)})")
