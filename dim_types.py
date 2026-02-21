# dim_types.py — Algebraic Type System for Dim
#
# Replaces string-keyed types ("int", "unknown") with a proper algebraic
# type representation that the type checker and MIR can operate on.

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple


# ── Primitive Types ────────────────────────────────────────────────────────────

class PrimKind(Enum):
    I8    = "i8"
    I16   = "i16"
    I32   = "i32"
    I64   = "i64"
    I128  = "i128"
    U8    = "u8"
    U16   = "u16"
    U32   = "u32"
    U64   = "u64"
    U128  = "u128"
    F32   = "f32"
    F64   = "f64"
    BOOL  = "bool"
    CHAR  = "char"
    STR   = "str"
    UNIT  = "()"     # zero-sized return type
    NEVER = "!"      # diverging (e.g. panic)


# ── Type ADT ──────────────────────────────────────────────────────────────────

class Type:
    """Base class for all Dim types."""

    def is_copy(self) -> bool:
        """Returns True if this type is implicitly copyable (no move semantics)."""
        return False

    def is_numeric(self) -> bool:
        return False

    def unify(self, other: "Type") -> Optional["Type"]:
        """
        Attempt unification with another type (basic HM unification).
        Returns the unified type or None on failure.
        """
        if isinstance(other, TypeVar) and not other.resolved:
            other.resolved = self
            return self
        if type(self) == type(other) and self == other:
            return self
        return None

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self).__name__)


@dataclass
class PrimType(Type):
    kind: PrimKind

    def is_copy(self) -> bool:
        return True  # All primitives are Copy

    def is_numeric(self) -> bool:
        return self.kind not in (PrimKind.BOOL, PrimKind.CHAR, PrimKind.STR,
                                  PrimKind.UNIT, PrimKind.NEVER)

    def unify(self, other: Type) -> Optional[Type]:
        if isinstance(other, TypeVar) and not other.resolved:
            other.resolved = self
            return self
        if isinstance(other, PrimType) and self.kind == other.kind:
            return self
        return None

    def __repr__(self) -> str:
        return self.kind.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PrimType) and self.kind == other.kind

    def __hash__(self):
        return hash(self.kind)


@dataclass
class TypeVar(Type):
    """An unresolved type variable (for Hindley-Milner inference)."""
    name: str
    resolved: Optional[Type] = field(default=None, compare=False, repr=False)

    def root(self) -> Type:
        """Follow resolution chain to the canonical type (path compression)."""
        if self.resolved is None:
            return self
        if isinstance(self.resolved, TypeVar):
            self.resolved = self.resolved.root()
        return self.resolved

    def unify(self, other: Type) -> Optional[Type]:
        root = self.root()
        if root is not self:
            return root.unify(other)
        other_root = other.root() if isinstance(other, TypeVar) else other
        if other_root is self:
            return self  # Already unified
        if isinstance(other_root, TypeVar):
            other_root.resolved = self
        else:
            self.resolved = other_root
        return other_root

    def __repr__(self) -> str:
        if self.resolved:
            return repr(self.resolved)
        return f"?{self.name}"


@dataclass
class RefType(Type):
    """&T or &mut T."""
    inner: Type
    mutable: bool = False

    def __repr__(self) -> str:
        mut = "mut " if self.mutable else ""
        return f"&{mut}{self.inner}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RefType) and self.inner == other.inner and self.mutable == other.mutable

    def __hash__(self):
        return hash(("&", self.mutable, self.inner))


@dataclass
class SliceType(Type):
    """[T] — unsized slice."""
    element: Type

    def __repr__(self) -> str:
        return f"[{self.element}]"


@dataclass
class ArrayType(Type):
    """[T; N] — fixed-size array."""
    element: Type
    size: int

    def is_copy(self) -> bool:
        return self.element.is_copy()

    def __repr__(self) -> str:
        return f"[{self.element}; {self.size}]"


@dataclass
class TupleType(Type):
    elements: List[Type]

    def is_copy(self) -> bool:
        return all(t.is_copy() for t in self.elements)

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.elements)
        return f"({inner})"


@dataclass
class FunctionType(Type):
    params: List[Type]
    return_type: Type
    is_async: bool = False
    is_effectful: bool = False

    def __repr__(self) -> str:
        prefix = "async " if self.is_async else ""
        params = ", ".join(repr(p) for p in self.params)
        return f"{prefix}fn({params}) -> {self.return_type}"


@dataclass
class GenericType(Type):
    """A generic/parameterised type application, e.g. Vec[i32], Option[str]."""
    name: str
    args: List[Type]

    def __repr__(self) -> str:
        args = ", ".join(repr(a) for a in self.args)
        return f"{self.name}[{args}]"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, GenericType)
                and self.name == other.name
                and self.args == other.args)

    def __hash__(self):
        return hash((self.name, tuple(self.args)))


@dataclass
class StructType(Type):
    name: str
    fields: Dict[str, Type] = field(default_factory=dict)
    generic_params: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return self.name


@dataclass
class EnumType(Type):
    name: str
    variants: Dict[str, Optional[List[Type]]] = field(default_factory=dict)
    generic_params: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return self.name


@dataclass
class TraitType(Type):
    name: str
    methods: Dict[str, FunctionType] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"trait {self.name}"


@dataclass
class TensorType(Type):
    """Native Tensor[dtype, shape] type for ML workloads."""
    dtype: PrimType
    shape: List[Optional[int]]   # None = dynamic dimension

    def __repr__(self) -> str:
        shape_str = ", ".join(str(d) if d is not None else "?" for d in self.shape)
        return f"Tensor[{self.dtype}, [{shape_str}]]"


@dataclass
class PromptType(Type):
    """Prompt[InputType, OutputType] — typed AI invocation."""
    input_type: Type
    output_type: Type
    deterministic: bool = False

    def __repr__(self) -> str:
        det = "deterministic " if self.deterministic else ""
        return f"{det}Prompt[{self.input_type}, {self.output_type}]"


@dataclass
class FutureType(Type):
    """Future[T] — result of an async computation."""
    inner: Type

    def __repr__(self) -> str:
        return f"Future[{self.inner}]"


@dataclass
class ResultType(Type):
    """Result[T, E] — fallible computation."""
    ok_type: Type
    err_type: Type

    def __repr__(self) -> str:
        return f"Result[{self.ok_type}, {self.err_type}]"


@dataclass
class OptionType(Type):
    """Option[T] — nullable value."""
    inner: Type

    def __repr__(self) -> str:
        return f"Option[{self.inner}]"


@dataclass
class UnknownType(Type):
    """Represents a type that couldn't be inferred — triggers E0031."""
    def __repr__(self) -> str:
        return "?"


# ── Type Aliases for common types ─────────────────────────────────────────────

I32   = PrimType(PrimKind.I32)
I64   = PrimType(PrimKind.I64)
F32   = PrimType(PrimKind.F32)
F64   = PrimType(PrimKind.F64)
BOOL  = PrimType(PrimKind.BOOL)
STR   = PrimType(PrimKind.STR)
UNIT  = PrimType(PrimKind.UNIT)
NEVER = PrimType(PrimKind.NEVER)

# ── Built-in type name resolution ─────────────────────────────────────────────

BUILTIN_TYPES: Dict[str, Type] = {
    "i8": PrimType(PrimKind.I8),
    "i16": PrimType(PrimKind.I16),
    "i32": I32,
    "i64": I64,
    "i128": PrimType(PrimKind.I128),
    "u8": PrimType(PrimKind.U8),
    "u16": PrimType(PrimKind.U16),
    "u32": PrimType(PrimKind.U32),
    "u64": PrimType(PrimKind.U64),
    "u128": PrimType(PrimKind.U128),
    "f32": F32,
    "f64": F64,
    "bool": BOOL,
    "char": PrimType(PrimKind.CHAR),
    "str": STR,
    "String": GenericType("String", []),
    "()": UNIT,
}


def resolve_builtin(name: str) -> Optional[Type]:
    return BUILTIN_TYPES.get(name)


def numeric_promotion(a: Type, b: Type) -> Optional[Type]:
    """
    Determine the result type of a numeric binary operation.
    Returns None if types are incompatible.
    """
    if not isinstance(a, PrimType) or not isinstance(b, PrimType):
        return None
    # Float dominates int
    float_kinds = {PrimKind.F32, PrimKind.F64}
    if a.kind in float_kinds or b.kind in float_kinds:
        if PrimKind.F64 in (a.kind, b.kind):
            return PrimType(PrimKind.F64)
        return PrimType(PrimKind.F32)
    # Wider integer wins
    int_order = [
        PrimKind.I8, PrimKind.I16, PrimKind.I32, PrimKind.I64, PrimKind.I128,
        PrimKind.U8, PrimKind.U16, PrimKind.U32, PrimKind.U64, PrimKind.U128,
    ]
    try:
        ia, ib = int_order.index(a.kind), int_order.index(b.kind)
        return PrimType(int_order[max(ia, ib)])
    except ValueError:
        return None
