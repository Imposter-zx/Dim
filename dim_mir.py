# dim_mir.py — Mid-Level Intermediate Representation (MIR) for Dim
#
# MIR is a typed, SSA-form, flat CFG representation used for:
#   - Borrow checking (operates on MIR, not AST)
#   - Optimization passes (DCE, constant folding, inlining)
#   - Code generation lowering to LLVM IR / MLIR

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Set, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from dim_types import Type


# ── Core data structures ────────────────────────────────────────────────────────

class Mutability(Enum):
    Mut = auto()
    Not = auto()


class BorrowKind(Enum):
    Shared  = auto()   # &T
    Mutable = auto()   # &mut T


@dataclass
class Local:
    """
    An SSA-like local variable slot.
    Every local is written exactly once (SSA property).
    """
    index:      int
    ty:         "Type"
    mutability: Mutability
    name:       Optional[str] = None    # Debug / source name

    def __repr__(self) -> str:
        pfx = "mut " if self.mutability == Mutability.Mut else ""
        nm  = self.name or f"_{self.index}"
        return f"Local({pfx}{nm}: {self.ty})"

    def __hash__(self):     return self.index
    def __eq__(self, o):    return isinstance(o, Local) and self.index == o.index


class ProjectionKind(Enum):
    Field      = auto()   # .field_name
    Index      = auto()   # [n]
    Deref      = auto()   # *ptr
    Downcast   = auto()   # as EnumVariant


@dataclass
class Projection:
    kind:  ProjectionKind
    value: Any   # field name (str), index (Local/int), variant name (str)

    def __repr__(self) -> str:
        if self.kind == ProjectionKind.Field:
            return f".{self.value}"
        if self.kind == ProjectionKind.Index:
            return f"[{self.value}]"
        if self.kind == ProjectionKind.Deref:
            return ".*"
        return f" as {self.value}"


@dataclass
class Place:
    """
    A memory location = a Local + optional projection chain.
    Examples:
      Local(x)           → x
      Local(s).Field("x") → s.x
      Local(v).Deref.Index(0) → (*v)[0]
    """
    local:       Local
    projections: List[Projection] = field(default_factory=list)
    ty:          Optional["Type"] = field(default=None, repr=False)

    def __repr__(self) -> str:
        projs = "".join(repr(p) for p in self.projections)
        return f"{self.local.name or f'_{self.local.index}'}{projs}"


# ── Operands ───────────────────────────────────────────────────────────────────

class Operand:
    pass


@dataclass
class ConstOperand(Operand):
    value: Any
    ty:    "Type"

    def __repr__(self) -> str:
        return f"Const({self.value}: {self.ty})"


@dataclass
class PlaceOperand(Operand):
    place: Place

    def __repr__(self) -> str:
        return f"Move({self.place})"


@dataclass
class BorrowOperand(Operand):
    kind:  BorrowKind
    place: Place

    def __repr__(self) -> str:
        pfx = "&mut " if self.kind == BorrowKind.Mutable else "&"
        return f"{pfx}{self.place}"


# ── RValues ────────────────────────────────────────────────────────────────────

class RValue:
    pass


@dataclass
class UseRValue(RValue):
    """Use (copy or move) an operand."""
    operand: Operand

    def __repr__(self) -> str:
        return repr(self.operand)


@dataclass
class BinOpRValue(RValue):
    op:    str   # "+", "-", "*", "/", "==", "!=", "<", ">", "<=", ">="
    left:  Operand
    right: Operand

    def __repr__(self) -> str:
        return f"{self.left} {self.op} {self.right}"


@dataclass
class UnOpRValue(RValue):
    op:      str   # "-", "!"
    operand: Operand

    def __repr__(self) -> str:
        return f"{self.op}{self.operand}"


@dataclass
class AggregateRValue(RValue):
    """Struct / tuple / array construction."""
    kind:   str   # "struct" | "tuple" | "array" | "enum"
    fields: List[Operand]
    ty:     "Type"


@dataclass
class CastRValue(RValue):
    operand: Operand
    ty:      "Type"

    def __repr__(self) -> str:
        return f"{self.operand} as {self.ty}"


# ── Statements ────────────────────────────────────────────────────────────────

class MIRStatement:
    pass


@dataclass
class Assign(MIRStatement):
    place:  Place
    rvalue: RValue

    def __repr__(self) -> str:
        return f"  {self.place} = {self.rvalue}"


@dataclass
class StorageLive(MIRStatement):
    """Allocate storage for a local (lifetime begins)."""
    local: Local

    def __repr__(self) -> str:
        return f"  StorageLive({self.local.name or self.local.index})"


@dataclass
class StorageDead(MIRStatement):
    """Free storage for a local (lifetime ends; Drop is called here)."""
    local: Local

    def __repr__(self) -> str:
        return f"  StorageDead({self.local.name or self.local.index})"


@dataclass
class Borrow(MIRStatement):
    """
    Create a borrow: dest = &[mut] place
    This is also where the borrow checker registers loans.
    """
    dest:  Local
    kind:  BorrowKind
    place: Place

    def __repr__(self) -> str:
        pfx = "&mut " if self.kind == BorrowKind.Mutable else "&"
        return f"  {self.dest.name} = {pfx}{self.place}"


@dataclass
class Drop(MIRStatement):
    """Explicit drop of a value (destructor call at end of scope)."""
    place: Place

    def __repr__(self) -> str:
        return f"  Drop({self.place})"


# ── Terminators ───────────────────────────────────────────────────────────────

class Terminator:
    pass


@dataclass
class Goto(Terminator):
    target: int  # BasicBlock id

    def __repr__(self) -> str:
        return f"  goto BB{self.target}"


@dataclass
class Branch(Terminator):
    condition:    Operand
    true_target:  int
    false_target: int

    def __repr__(self) -> str:
        return f"  branch {self.condition} → BB{self.true_target}, BB{self.false_target}"


@dataclass
class SwitchInt(Terminator):
    """match-like multi-way branch."""
    discriminant: Operand
    targets:      Dict[int, int]    # value → block_id
    otherwise:    Optional[int]


@dataclass
class Return(Terminator):
    value: Optional[Operand]

    def __repr__(self) -> str:
        return f"  return {self.value}"


@dataclass
class Call(Terminator):
    func:       Operand
    args:       List[Operand]
    dest:       Optional[Place]
    next_block: int
    unwind:     Optional[int] = None  # block to jump to on panic

    def __repr__(self) -> str:
        dest_s = f"{self.dest} = " if self.dest else ""
        args_s = ", ".join(repr(a) for a in self.args)
        return f"  {dest_s}call {self.func}({args_s}) → BB{self.next_block}"


@dataclass
class PromptCall(Terminator):
    """
    Native terminator for LLM inference.
    Suspends execution pending the model response.
    """
    prompt_ref:   str
    input:        Operand
    dest:         Optional[Place]
    next_block:   int
    deterministic: bool = False

    def __repr__(self) -> str:
        det = "det " if self.deterministic else ""
        return f"  {self.dest} = {det}prompt_call {self.prompt_ref}({self.input}) → BB{self.next_block}"


@dataclass
class Panic(Terminator):
    message: str

    def __repr__(self) -> str:
        return f"  panic({self.message!r})"


@dataclass
class Unreachable(Terminator):
    def __repr__(self) -> str:
        return "  unreachable"


# ── Basic Block ────────────────────────────────────────────────────────────────

@dataclass
class BasicBlock:
    id:         int
    stmts:      List[MIRStatement] = field(default_factory=list)
    terminator: Optional[Terminator] = None
    # CFG edges (populated during CFG construction)
    predecessors: List[int] = field(default_factory=list, compare=False, repr=False)

    def successors(self) -> List[int]:
        t = self.terminator
        if isinstance(t, Goto):             return [t.target]
        if isinstance(t, Branch):           return [t.true_target, t.false_target]
        if isinstance(t, (Call, PromptCall)): return [t.next_block]
        if isinstance(t, SwitchInt):
            out = list(t.targets.values())
            if t.otherwise is not None:
                out.append(t.otherwise)
            return out
        return []

    def __repr__(self) -> str:
        lines = [f"BB{self.id}:"]
        for s in self.stmts:
            lines.append(repr(s))
        if self.terminator:
            lines.append(repr(self.terminator))
        return "\n".join(lines)


# ── MIR Function ──────────────────────────────────────────────────────────────

@dataclass
class MIRFunction:
    name:            str
    params:          List[Local]
    return_type:     "Type"
    locals:          Dict[int, Local]
    blocks:          List[BasicBlock]
    lifetime_params: List[str] = field(default_factory=list)
    is_async:        bool = False

    def entry_block(self) -> BasicBlock:
        return self.blocks[0]

    def build_predecessors(self):
        """Populate predecessor lists for all blocks."""
        for bb in self.blocks:
            bb.predecessors.clear()
        for bb in self.blocks:
            for succ_id in bb.successors():
                self.blocks[succ_id].predecessors.append(bb.id)

    def pretty(self) -> str:
        lines = [f"fn {self.name}(", ]
        for p in self.params:
            lines.append(f"  {p},")
        lines.append(f") -> {self.return_type}:")
        for bb in self.blocks:
            lines.append(repr(bb))
            lines.append("")
        return "\n".join(lines)


@dataclass
class MIRModule:
    name:      str
    functions: List[MIRFunction] = field(default_factory=list)
    globals:   Dict[str, Any]    = field(default_factory=dict)

    def get_fn(self, name: str) -> Optional[MIRFunction]:
        for f in self.functions:
            if f.name == name:
                return f
        return None


# ── Helper: CFG Utilities ──────────────────────────────────────────────────────

def cfg_dominators(fn: MIRFunction) -> Dict[int, Set[int]]:
    """
    Compute the dominator set for each block using the iterative dataflow
    algorithm (Cooper et al.).
    dom[n] = {n} ∪ (∩ dom[pred] for pred in predecessors(n))
    """
    fn.build_predecessors()
    n = len(fn.blocks)
    dom: Dict[int, Set[int]] = {i: set(range(n)) for i in range(n)}
    dom[0] = {0}
    changed = True
    while changed:
        changed = False
        for i in range(1, n):
            bb = fn.blocks[i]
            if not bb.predecessors:
                continue
            new_dom = set.intersection(*(dom[p] for p in bb.predecessors)) | {i}
            if new_dom != dom[i]:
                dom[i] = new_dom
                changed = True
    return dom


def cfg_liveness(fn: MIRFunction) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """
    Compute LiveIn and LiveOut sets for each basic block.
    Used by the borrow checker to determine when locals die.
    Returns: (live_in, live_out) where keys are block ids and values are sets of local indices.
    """
    def uses_of(bb: BasicBlock) -> Set[int]:
        used: Set[int] = set()
        for s in bb.stmts:
            if isinstance(s, Assign):
                _collect_operand_uses(s.rvalue, used)
        if bb.terminator:
            _collect_terminator_uses(bb.terminator, used)
        return used

    def defs_of(bb: BasicBlock) -> Set[int]:
        defs: Set[int] = set()
        for s in bb.stmts:
            if isinstance(s, Assign):
                defs.add(s.place.local.index)
            elif isinstance(s, Borrow):
                defs.add(s.dest.index)
        return defs

    fn.build_predecessors()
    live_in:  Dict[int, Set[int]] = {bb.id: set() for bb in fn.blocks}
    live_out: Dict[int, Set[int]] = {bb.id: set() for bb in fn.blocks}
    changed = True
    while changed:
        changed = False
        for bb in reversed(fn.blocks):
            out = set()
            for succ_id in bb.successors():
                out |= live_in[succ_id]
            new_in = uses_of(bb) | (out - defs_of(bb))
            if new_in != live_in[bb.id] or out != live_out[bb.id]:
                live_in[bb.id]  = new_in
                live_out[bb.id] = out
                changed = True
    return live_in, live_out


def _collect_operand_uses(rval: Any, out: Set[int]):
    if isinstance(rval, UseRValue):
        _op_use(rval.operand, out)
    elif isinstance(rval, BinOpRValue):
        _op_use(rval.left, out)
        _op_use(rval.right, out)
    elif isinstance(rval, UnOpRValue):
        _op_use(rval.operand, out)


def _op_use(op: Operand, out: Set[int]):
    if isinstance(op, (PlaceOperand, BorrowOperand)):
        out.add(op.place.local.index)


def _collect_terminator_uses(t: Terminator, out: Set[int]):
    if isinstance(t, Branch):
        _op_use(t.condition, out)
    elif isinstance(t, Call):
        _op_use(t.func, out)
        for a in t.args:
            _op_use(a, out)
    elif isinstance(t, Return) and t.value:
        _op_use(t.value, out)
    elif isinstance(t, PromptCall):
        _op_use(t.input, out)
