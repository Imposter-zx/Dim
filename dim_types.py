# dim_types.py — Algebraic Type System for Dim
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple


@dataclass
class Type:
    def unify(self, other):
        # type: (Type) -> Optional[Type]
        if self == other:
            return self
        if isinstance(other, UnknownType):
            return self
        if isinstance(self, UnknownType):
            return other
        return None


@dataclass
class PrimType(Type):
    kind: str

    def __repr__(self):
        return self.kind

    def __hash__(self):
        return hash(self.kind)


I32 = PrimType("i32")
I64 = PrimType("i64")
F32 = PrimType("f32")
F64 = PrimType("f64")
BOOL = PrimType("bool")
STR = PrimType("str")
UNIT = PrimType("Unit")
NEVER = PrimType("Never")


@dataclass
class TypeVar(Type):
    name: str

    def __repr__(self):
        return "'" + self.name

    def __hash__(self):
        return hash(self.name)


@dataclass
class FunctionType(Type):
    params: List[Type]
    return_type: Type
    is_async: bool = False
    capabilities: List[str] = field(default_factory=list)

    def __repr__(self):
        p_str = ", ".join(repr(p) for p in self.params)
        async_str = "async " if self.is_async else ""
        cap_str = " caps:" + str(self.capabilities) if self.capabilities else ""
        return async_str + "fn(" + p_str + ") -> " + repr(self.return_type) + cap_str


@dataclass
class TensorType(Type):
    dtype: Type
    shape: List[Optional[int]]

    def __repr__(self):
        shape_str = ", ".join(str(d) if d is not None else "?" for d in self.shape)
        return f"Tensor[{self.dtype}, [{shape_str}]]"

    def shape_rank(self) -> int:
        return len(self.shape)

    def is_fully_static(self) -> bool:
        return all(d is not None for d in self.shape)

    def unify_shapes(self, other: "TensorType") -> Optional[List[Optional[int]]]:
        if len(self.shape) != len(other.shape):
            return None
        result: List[Optional[int]] = []
        for a, b in zip(self.shape, other.shape):
            if a is not None and b is not None:
                if a != b:
                    return None
                result.append(a)
            elif a is not None:
                result.append(a)
            else:
                result.append(b)
        return result


@dataclass
class SymbolicDim:
    name: str
    constraints: List[Tuple[str, int]] = field(default_factory=list)

    def __repr__(self):
        if self.constraints:
            cons = ", ".join(f"{op}{val}" for op, val in self.constraints)
            return f"{self.name}::{cons}"
        return self.name

    def constrain(self, op: str, val: int):
        self.constraints.append((op, val))


@dataclass
class PromptType(Type):
    name: str
    input_type: Type
    output_type: Type
    deterministic: bool = False

    def __repr__(self):
        return "Prompt<" + self.name + ">"


@dataclass
class GenericType(Type):
    name: str
    args: List[Type]

    def __repr__(self):
        args_str = ", ".join(repr(a) for a in self.args)
        return self.name + "[" + args_str + "]"


@dataclass
class RefType(Type):
    inner: Type
    mutable: bool = False

    def __repr__(self):
        m = "mut " if self.mutable else ""
        return f"&{m}{self.inner}"

    def __hash__(self):
        return hash(("RefType", self.inner, self.mutable))


@dataclass
class StructType(Type):
    name: str
    fields: Dict[str, Type] = field(default_factory=dict)

    def __repr__(self):
        return self.name


@dataclass
class EnumType(Type):
    name: str
    variants: Dict[str, Optional[List[Type]]] = field(default_factory=dict)

    def __repr__(self):
        return self.name


@dataclass
class FutureType(Type):
    inner: Type

    def __repr__(self):
        return "Future[" + repr(self.inner) + "]"


@dataclass
class UnknownType(Type):
    def __repr__(self):
        return "?"


BUILTIN_TYPES = {
    "i32": I32,
    "i64": I64,
    "f32": F32,
    "f64": F64,
    "bool": BOOL,
    "str": STR,
    "Unit": UNIT,
    "Never": NEVER,
}


def resolve_builtin(name):
    return BUILTIN_TYPES.get(name)


def numeric_promotion(t1, t2):
    NUMERIC_TYPES = {I32, I64, F32, F64}
    if t1 not in NUMERIC_TYPES or t2 not in NUMERIC_TYPES:
        return None
    if t1 == F64 or t2 == F64:
        return F64
    if t1 == F32 or t2 == F32:
        return F32
    if t1 == I64 or t2 == I64:
        return I64
    return I32
