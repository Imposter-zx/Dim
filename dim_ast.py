# dim_ast.py — Span-annotated AST for Dim (v0.2)
#
# Every node carries an optional Span for precise error reporting.
# Types use dim_types.Type (algebraic ADT) instead of strings.
#
# Design: Node is NOT a @dataclass to avoid the Python dataclass
# inheritance problem (non-default field after default field).
# Each concrete @dataclass declares `span: Optional[Span] = None`
# as its LAST field so the parser can pass span=... as a keyword arg.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union, TYPE_CHECKING

from dim_token import Span

if TYPE_CHECKING:
    from dim_types import Type


# ── Base classes (NOT @dataclass) ─────────────────────────────────────────────

class Node:
    """Base for all AST nodes."""
    def accept(self, visitor: "Visitor"):
        method = f"visit_{type(self).__name__}"
        handler = getattr(visitor, method, visitor.generic_visit)
        return handler(self)


class Statement(Node):
    """Base for statement nodes."""
    pass


class Expression(Node):
    """Base for expression nodes. resolved_type filled by type checker."""
    resolved_type: Optional["Type"] = None


# ── Expressions ───────────────────────────────────────────────────────────────

@dataclass
class Literal(Expression):
    value: Union[int, float, str, bool, None]
    span:  Optional[Span] = None

    def __repr__(self) -> str:
        return f"Literal({self.value!r})"


@dataclass
class Identifier(Expression):
    name: str
    span: Optional[Span] = None

    def __repr__(self) -> str:
        return f"Identifier({self.name!r})"


@dataclass
class BinaryOp(Expression):
    left:  Expression
    op:    str
    right: Expression
    span:  Optional[Span] = None

    def __repr__(self) -> str:
        return f"BinaryOp({self.left}, {self.op!r}, {self.right})"


@dataclass
class UnaryOp(Expression):
    op:      str
    operand: Expression
    span:    Optional[Span] = None


@dataclass
class Call(Expression):
    callee: Expression
    args:   List[Expression]
    kwargs: List[tuple] = field(default_factory=list)
    span:   Optional[Span] = None


@dataclass
class MethodCall(Expression):
    receiver: Expression
    method:   str
    args:     List[Expression]
    span:     Optional[Span] = None


@dataclass
class MemberAccess(Expression):
    object: Expression
    member: str
    span:   Optional[Span] = None


@dataclass
class IndexAccess(Expression):
    object: Expression
    index:  Expression
    span:   Optional[Span] = None


@dataclass
class BorrowExpr(Expression):
    """&expr or &mut expr"""
    expr:    Expression
    mutable: bool = False
    span:    Optional[Span] = None


@dataclass
class DerefExpr(Expression):
    """*expr"""
    expr: Expression
    span: Optional[Span] = None


@dataclass
class AwaitExpr(Expression):
    """await expr"""
    expr: Expression
    span: Optional[Span] = None


@dataclass
class CastExpr(Expression):
    """expr as Type"""
    expr:    Expression
    cast_to: "Type"
    span:    Optional[Span] = None


@dataclass
class ListLiteral(Expression):
    elements: List[Expression]
    span:     Optional[Span] = None


@dataclass
class TupleLiteral(Expression):
    elements: List[Expression]
    span:     Optional[Span] = None


@dataclass
class StructLiteral(Expression):
    name:   str
    fields: List[tuple]        # [(field_name, expr)]
    span:   Optional[Span] = None


@dataclass
class ClosureExpr(Expression):
    params:      List[tuple]           # [(name, Optional[Type])]
    return_type: Optional["Type"]
    body:        List["Statement"]
    is_async:    bool = False
    span:        Optional[Span] = None


@dataclass
class IfExpr(Expression):
    """if/else as expression."""
    condition: Expression
    then_expr: Expression
    else_expr: Expression
    span:      Optional[Span] = None


@dataclass
class ModelCall(Expression):
    """Native LLM invocation."""
    model_name: str
    input:      Expression
    prompt:     Optional[str] = None
    span:       Optional[Span] = None


@dataclass
class TensorExpr(Expression):
    dtype: str
    shape: List[int]
    op:    str
    span:  Optional[Span] = None


# ── Statements ────────────────────────────────────────────────────────────────

@dataclass
class ExprStmt(Statement):
    expr: Expression
    span: Optional[Span] = None


@dataclass
class LetStmt(Statement):
    name:     str
    is_mut:   bool
    type_ann: Optional["Type"]
    value:    Expression
    span:     Optional[Span] = None


@dataclass
class AssignStmt(Statement):
    target: Expression
    op:     str
    value:  Expression
    span:   Optional[Span] = None


@dataclass
class ReturnStmt(Statement):
    value: Optional[Expression] = None
    span:  Optional[Span] = None


@dataclass
class BreakStmt(Statement):
    label: Optional[str] = None
    span:  Optional[Span] = None


@dataclass
class ContinueStmt(Statement):
    label: Optional[str] = None
    span:  Optional[Span] = None


@dataclass
class IfStmt(Statement):
    condition:     Expression
    then_branch:   List[Statement]
    elif_branches: List[tuple] = field(default_factory=list)
    else_branch:   Optional[List[Statement]] = None
    span:          Optional[Span] = None


@dataclass
class WhileStmt(Statement):
    condition: Expression
    body:      List[Statement]
    label:     Optional[str] = None
    span:      Optional[Span] = None


@dataclass
class ForStmt(Statement):
    iterator: str
    iterable: Expression
    body:     List[Statement]
    label:    Optional[str] = None
    span:     Optional[Span] = None


@dataclass
class MatchArm(Node):
    pattern: Expression
    guard:   Optional[Expression]
    body:    List[Statement]
    span:    Optional[Span] = None


@dataclass
class MatchStmt(Statement):
    expression: Expression
    arms:       List[MatchArm]
    span:       Optional[Span] = None


@dataclass
class UnsafeBlock(Statement):
    body: List[Statement]
    span: Optional[Span] = None


# ── Top-level Declarations ────────────────────────────────────────────────────

@dataclass
class Param(Node):
    name:     str
    type_ann: Optional["Type"] = None
    is_mut:   bool = False
    is_self:  bool = False
    span:     Optional[Span] = None


@dataclass
class FunctionDef(Statement):
    name:        str
    params:      List[Param]
    return_type: Optional["Type"]
    body:        List[Statement]
    is_async:    bool = False
    is_pub:      bool = False
    generics:    List[str] = field(default_factory=list)
    resolved_fn_type: Optional["Type"] = field(default=None, compare=False, repr=False)
    span:        Optional[Span] = None


@dataclass
class StructDef(Statement):
    name:     str
    fields:   List[tuple]    # [(name, Type, is_pub)]
    generics: List[str] = field(default_factory=list)
    is_pub:   bool = False
    span:     Optional[Span] = None


@dataclass
class EnumDef(Statement):
    name:     str
    variants: List[tuple]    # [(name, Optional[List[Type]])]
    generics: List[str] = field(default_factory=list)
    is_pub:   bool = False
    span:     Optional[Span] = None


@dataclass
class TraitDef(Statement):
    name:     str
    methods:  List[FunctionDef]
    generics: List[str] = field(default_factory=list)
    is_pub:   bool = False
    span:     Optional[Span] = None


@dataclass
class ImplBlock(Statement):
    """impl Trait for Type  or  impl Type"""
    type_name:  str
    trait_name: Optional[str]
    methods:    List[FunctionDef]
    generics:   List[str] = field(default_factory=list)
    span:       Optional[Span] = None


@dataclass
class PromptRole(Node):
    role_name: str
    content:   str
    span:      Optional[Span] = None


@dataclass
class PromptDef(Statement):
    name:          str
    base:          Optional[str]
    roles:         List[PromptRole]
    input_type:    Optional["Type"]
    output_type:   Optional["Type"]
    deterministic: bool = False
    is_pub:        bool = False
    span:          Optional[Span] = None


@dataclass
class ActorDef(Statement):
    name:     str
    fields:   List[tuple]           # [(name, Type, Optional[default_expr])]
    handlers: List[FunctionDef]
    is_pub:   bool = False
    span:     Optional[Span] = None


@dataclass
class ImportStmt(Statement):
    path:  List[str]
    names: Optional[List[str]]
    alias: Optional[str] = None
    span:  Optional[Span] = None


@dataclass
class TypeAlias(Statement):
    name:     str
    alias_of: "Type"
    generics: List[str] = field(default_factory=list)
    span:     Optional[Span] = None


@dataclass
class Program(Node):
    statements: List[Statement]
    span:       Optional[Span] = None


# ── Visitor Pattern ────────────────────────────────────────────────────────────

class Visitor:
    """
    Extensible visitor base.  Override visit_* methods, or override visit()
    for a catch-all dispatch.
    """

    def visit(self, node: Node):
        method = f"visit_{type(node).__name__}"
        handler = getattr(self, method, self.generic_visit)
        return handler(node)

    def generic_visit(self, node: Node):
        """Default: recurse into all Node-typed fields."""
        for val in vars(node).values():
            if isinstance(val, Node):
                self.visit(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, Node):
                        self.visit(item)
