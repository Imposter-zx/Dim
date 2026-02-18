# Dim AST Node Definitions (POC)

from dataclasses import dataclass, field
from typing import List, Optional, Union

@dataclass
class Node:
    pass

@dataclass
class Expression(Node):
    pass

@dataclass
class Statement(Node):
    pass

@dataclass
class ReturnStmt(Statement):
    value: Optional[Expression]

@dataclass
class Literal(Expression):
    value: Union[int, float, str, bool]

@dataclass
class Identifier(Expression):
    name: str

@dataclass
class BinaryOp(Expression):
    left: Expression
    op: str
    right: Expression

@dataclass
class Call(Expression):
    callee: Expression
    args: List[Expression]

@dataclass
class LetStmt(Statement):
    name: str
    is_mut: bool
    type_ann: Optional[str]
    value: Expression

@dataclass
class FunctionDef(Statement):
    name: str
    params: List[tuple] # (name, type)
    return_type: Optional[str]
    body: List[Statement]

@dataclass
class IfStmt(Statement):
    condition: Expression
    then_branch: List[Statement]
    else_branch: Optional[List[Statement]]

@dataclass
class WhileStmt(Statement):
    condition: Expression
    body: List[Statement]

@dataclass
class ForStmt(Statement):
    iterator: str
    iterable: Expression
    body: List[Statement]

@dataclass
class StructDef(Statement):
    name: str
    fields: List[tuple] # (name, type)

@dataclass
class EnumDef(Statement):
    name: str
    variants: List[tuple] # (name, fields_or_none)

@dataclass
class TraitDef(Statement):
    name: str
    methods: List[FunctionDef]

@dataclass
class MatchArm(Node):
    pattern: Expression # Simplified
    condition: Optional[Expression]
    body: List[Statement]

@dataclass
class MatchStmt(Statement):
    expression: Expression
    arms: List[MatchArm]

@dataclass
class PromptDef(Statement):
    name: str
    base: Optional[str]
    roles: List[tuple] # (role_name, content)
    output_type: Optional[str]

@dataclass
class ModelCall(Expression):
    model_name: str
    input: Expression
    prompt: Optional[str]

@dataclass
class TensorExpr(Expression):
    dtype: str
    shape: List[int]
    op: str # "ones", "zeros", etc.

@dataclass
class Program(Node):
    statements: List[Statement]
