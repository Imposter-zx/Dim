# dim_ast.py — Span-annotated AST for Dim
from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple, TYPE_CHECKING
from dim_token import Span

if TYPE_CHECKING:
    from dim_types import Type


class Node:
    def accept(self, visitor):
        method = "visit_" + type(self).__name__
        handler = getattr(visitor, method, visitor.generic_visit)
        return handler(self)


class Statement(Node):
    pass


class Expression(Node):
    resolved_type = None


@dataclass
class Literal(Expression):
    value: Union[int, float, str, bool, None]
    span: Optional[Span] = None


@dataclass
class Identifier(Expression):
    name: str
    span: Optional[Span] = None


@dataclass
class BinaryOp(Expression):
    left: Expression
    op: str
    right: Expression
    span: Optional[Span] = None


@dataclass
class UnaryOp(Expression):
    op: str
    operand: Expression
    span: Optional[Span] = None


@dataclass
class Call(Expression):
    callee: Expression
    args: List[Expression]
    kwargs: List[tuple] = field(default_factory=list)
    span: Optional[Span] = None


@dataclass
class MethodCall(Expression):
    receiver: Expression
    method: str
    args: List[Expression]
    span: Optional[Span] = None


@dataclass
class BorrowExpr(Expression):
    expr: Expression
    mutable: bool = False
    span: Optional[Span] = None


@dataclass
class DerefExpr(Expression):
    expr: Expression
    span: Optional[Span] = None


@dataclass
class AwaitExpr(Expression):
    expr: Expression
    span: Optional[Span] = None


@dataclass
class ListLiteral(Expression):
    elements: List[Expression]
    span: Optional[Span] = None


@dataclass
class TensorExpr(Expression):
    dtype: str
    shape: List[int]
    op: str = "const"
    span: Optional[Span] = None


@dataclass
class ModelCall(Expression):
    model_name: str
    input: Expression
    prompt: Optional[str] = None
    span: Optional[Span] = None


@dataclass
class ExprStmt(Statement):
    expr: Expression
    span: Optional[Span] = None


@dataclass
class LetStmt(Statement):
    name: str
    is_mut: bool
    type_ann: Optional["Type"]
    value: Expression
    span: Optional[Span] = None


@dataclass
class AssignStmt(Statement):
    target: Expression
    op: str
    value: Expression
    span: Optional[Span] = None


@dataclass
class ReturnStmt(Statement):
    value: Optional[Expression] = None
    span: Optional[Span] = None


@dataclass
class IfStmt(Statement):
    condition: Expression
    then_branch: List[Statement]
    elif_branches: List[tuple] = field(default_factory=list)
    else_branch: Optional[List[Statement]] = None
    span: Optional[Span] = None


@dataclass
class WhileStmt(Statement):
    condition: Expression
    body: List[Statement]
    span: Optional[Span] = None


@dataclass
class ForStmt(Statement):
    iterator: str
    iterable: Expression
    body: List[Statement]
    span: Optional[Span] = None


@dataclass
class Param(Node):
    name: str
    type_ann: Optional["Type"] = None
    is_mut: bool = False
    span: Optional[Span] = None


@dataclass
class FunctionDef(Statement):
    name: str
    params: List[Param]
    return_type: Optional["Type"]
    body: List[Statement]
    is_async: bool = False
    capabilities: List[str] = field(default_factory=list)
    resolved_fn_type: Optional["Type"] = None
    tool: Optional["ToolDecorator"] = None
    generics: List[str] = field(default_factory=list)
    span: Optional[Span] = None


@dataclass
class PromptRole(Node):
    role_name: str
    content: str
    span: Optional[Span] = None


@dataclass
class PromptDef(Statement):
    name: str
    base: Optional[str]
    roles: List[PromptRole]
    input_type: Optional["Type"]
    output_type: Optional["Type"]
    deterministic: bool = False
    span: Optional[Span] = None


@dataclass
class ToolDecorator(Node):
    name: str
    permissions: List[str]
    description: Optional[str] = None
    span: Optional[Span] = None


@dataclass
class ActorDef(Statement):
    name: str
    fields: List[Tuple]
    handlers: List[FunctionDef]
    span: Optional[Span] = None


@dataclass
class ImportStmt(Statement):
    path: List[str]
    module: Optional[str] = None
    alias: Optional[str] = None
    span: Optional[Span] = None


@dataclass
class StructDef(Statement):
    name: str
    fields: List[Tuple]
    generics: List[str] = field(default_factory=list)
    span: Optional[Span] = None


@dataclass
class EnumDef(Statement):
    name: str
    variants: List[Tuple]
    generics: List[str] = field(default_factory=list)
    span: Optional[Span] = None


@dataclass
class TraitDef(Statement):
    name: str
    methods: List[FunctionDef]
    generics: List[str] = field(default_factory=list)
    span: Optional[Span] = None


@dataclass
class ImplBlock(Statement):
    name: str
    trait_name: Optional[str]
    methods: List[FunctionDef]
    generics: List[str] = field(default_factory=list)
    span: Optional[Span] = None


@dataclass
class MatchArm(Node):
    pattern: Expression
    guard: Optional[Expression] = None
    body: List[Statement] = field(default_factory=list)
    span: Optional[Span] = None


@dataclass
class MatchStmt(Statement):
    expr: Expression
    arms: List[MatchArm]
    span: Optional[Span] = None


@dataclass
class BreakStmt(Statement):
    span: Optional[Span] = None


@dataclass
class ContinueStmt(Statement):
    span: Optional[Span] = None


@dataclass
class UnsafeBlock(Statement):
    body: List[Statement]
    span: Optional[Span] = None


@dataclass
class Program(Node):
    statements: List[Statement]
    span: Optional[Span] = None


class Visitor:
    def visit(self, node):
        method = "visit_" + type(node).__name__
        handler = getattr(self, method, self.generic_visit)
        return handler(node)

    def generic_visit(self, node):
        for val in vars(node).values():
            if isinstance(val, Node):
                self.visit(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, Node):
                        self.visit(item)
