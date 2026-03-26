# dim_mir.py — Mid-Level IR for Dim
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, Any
from enum import Enum, auto


class Mutability(Enum):
    Not = auto()
    Mut = auto()


class BorrowKind(Enum):
    Shared = auto()  # &T
    Mutable = auto()  # &mut T


@dataclass
class Local:
    index: int
    name: Optional[str] = None
    ty: Any = None  # dim_types.Type
    mutability: Mutability = Mutability.Not

    def __post_init__(self):
        if not isinstance(self.mutability, Mutability):
            self.mutability = Mutability.Not


@dataclass
class Place:
    local: "Local"
    projection: List[str] = field(default_factory=list)


class Operand:
    pass


@dataclass
class ConstOperand(Operand):
    ty: Any
    value: Any


@dataclass
class PlaceOperand(Operand):
    place: Place


@dataclass
class BorrowOperand(Operand):
    place: Place
    kind: BorrowKind


class RValue:
    pass


@dataclass
class UseRValue(RValue):
    operand: Operand


@dataclass
class BinOpRValue(RValue):
    left: Operand
    op: str
    right: Operand


@dataclass
class UnOpRValue(RValue):
    op: str
    operand: Operand


@dataclass
class TensorRValue(RValue):
    dtype: Any
    shape: List[int]
    op: str = "const"


class MIRStatement:
    pass


@dataclass
class Assign(MIRStatement):
    place: Place
    rvalue: RValue


@dataclass
class StorageLive(MIRStatement):
    local: Local


@dataclass
class StorageDead(MIRStatement):
    local: Local


@dataclass
class Borrow(MIRStatement):
    dest: Place
    kind: BorrowKind
    place: Place


@dataclass
class Drop(MIRStatement):
    place: Place


class Terminator:
    pass


@dataclass
class Return(Terminator):
    value: Optional[Operand] = None


@dataclass
class Goto(Terminator):
    target: int


@dataclass
class Branch(Terminator):
    condition: Operand
    true_target: int
    false_target: int


@dataclass
class Call(Terminator):
    callee: Any
    args: List[Operand]
    dest: Optional[Place]
    next_block: "BasicBlock"


@dataclass
class PromptCall(Terminator):
    prompt_ref: str
    input: Operand
    dest: Optional[Place]
    next_block: "BasicBlock"
    deterministic: bool = False


@dataclass
class BasicBlock:
    id: int
    stmts: List[MIRStatement] = field(default_factory=list)
    terminator: Optional[Terminator] = None


@dataclass
class MIRFunction:
    name: str
    params: List[Local]
    return_type: Any
    blocks: List[BasicBlock] = field(default_factory=list)
    locals: List[Local] = field(default_factory=list)
    locals_map: Dict[int, Local] = field(default_factory=dict)

    def __post_init__(self):
        self.locals_map = {loc.index: loc for loc in self.locals}

    def pretty(self) -> str:
        lines = [
            f"fn {self.name}({', '.join(p.name or f'%{p.index}' for p in self.params)}) -> {self.return_type}:"
        ]
        for bb in self.blocks:
            lines.append(f"  bb{bb.id}:")
            for s in bb.stmts:
                lines.append(f"    {s}")
            if bb.terminator:
                lines.append(f"    → {bb.terminator}")
        return "\n".join(lines)


@dataclass
class MIRModule:
    name: str
    functions: List[MIRFunction] = field(default_factory=list)


def cfg_liveness(fn: MIRFunction) -> tuple:
    """
    Simple liveness analysis for MIR basic blocks.
    Returns (live_in, live_out) dicts: block object id -> set of local indices.
    """
    live_in: Dict[int, set] = {id(bb): set() for bb in fn.blocks}
    live_out: Dict[int, set] = {id(bb): set() for bb in fn.blocks}

    def block_locals(bb: BasicBlock) -> set:
        result = set()
        for s in bb.stmts:
            if isinstance(s, StorageLive):
                result.add(s.local.index)
            elif isinstance(s, Assign):
                if isinstance(s.place, Place):
                    result.add(s.place.local.index)
                elif hasattr(s.place, "local"):
                    result.add(s.place.local.index)
        return result

    changed = True
    while changed:
        changed = False
        for bb in fn.blocks:
            out = set()
            for succ_bb in _successors_bb(bb, fn):
                out |= live_in.get(id(succ_bb), set())
            live_out[id(bb)] = out

            kill = block_locals(bb)
            inp = (live_out[id(bb)] - kill) | kill
            if inp != live_in[id(bb)]:
                live_in[id(bb)] = inp
                changed = True

    return live_in, live_out


def _successors(bb: BasicBlock, fn: MIRFunction) -> List[int]:
    term = bb.terminator
    if term is None:
        return []
    if isinstance(term, Goto):
        return [term.target]
    if isinstance(term, Branch):
        return [term.true_target, term.false_target]
    if isinstance(term, Return):
        return []
    if isinstance(term, (Call, PromptCall)):
        return [term.next_block.id]


def _successors_bb(bb: BasicBlock, fn: MIRFunction) -> List[BasicBlock]:
    term = bb.terminator
    if term is None:
        return []
    if isinstance(term, Goto):
        for b in fn.blocks:
            if b.id == term.target:
                return [b]
        return []
    if isinstance(term, Branch):
        result = []
        for b in fn.blocks:
            if b.id == term.true_target:
                result.append(b)
            elif b.id == term.false_target:
                result.append(b)
        return result
    if isinstance(term, Return):
        return []
    if isinstance(term, (Call, PromptCall)):
        return [term.next_block]
    return []
    return []
