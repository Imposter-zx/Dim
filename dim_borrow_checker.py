# dim_borrow_checker.py — Ownership & Borrow Checker for Dim
#
# Operates on MIR (not AST) after the lowering pass.
# Implements Polonius-inspired loan invalidation analysis:
#   1. Track ownership moves
#   2. Issue loans for borrows
#   3. Check loan invalidation at mutation/move points
#   4. Check loans don't outlive their owners

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from dim_mir import (
    MIRFunction, BasicBlock, Local, Place, Mutability, BorrowKind,
    MIRStatement, Assign, StorageLive, StorageDead, Borrow, Drop,
    Terminator, Goto, Branch, Call, PromptCall, Return,
    UseRValue, BorrowOperand, PlaceOperand,
    cfg_liveness,
)
from dim_diagnostic import DiagnosticBag, Severity
from dim_token import Span


class MoveState(Enum):
    Owned    = auto()   # Value is here and owned
    Moved    = auto()   # Value has been moved out → use is an error
    Borrowed = auto()   # Value is currently borrowed


@dataclass
class Loan:
    id:         int
    kind:       BorrowKind
    place:      Place          # The borrowed place
    issued_at:  Tuple[int, int]  # (block_id, stmt_index)
    region:     Set[Tuple[int, int]]  # Points where this loan is live


class BorrowChecker:
    """
    Dim borrow checker operating on a single MIRFunction.

    Error codes emitted:
      E0040 — use of moved value
      E0041 — mutable borrow while already borrowed
      E0042 — shared borrow while mutably borrowed
      E0043 — dangling reference (borrow outlives owner)
      E0044 — assignment to immutable binding
    """

    def __init__(self, fn: MIRFunction, diag: DiagnosticBag):
        self.fn   = fn
        self.diag = diag
        self._loan_counter = 0
        # Per-block state (simplified linear walk for Phase 1)
        self.move_state: Dict[int, MoveState] = {}   # local.index → state
        self.active_loans: Dict[int, Loan]    = {}   # loan_id → Loan
        # Loans indexed by the local they borrow
        self.loans_on: Dict[int, List[int]]   = {}   # local.index → [loan_ids]

    # ── Entry ──────────────────────────────────────────────────────────────────

    def check(self):
        """Run the borrow checker over the entire function."""
        # Initialise all params as Owned
        for p in self.fn.params:
            self.move_state[p.index] = MoveState.Owned

        live_in, live_out = cfg_liveness(self.fn)

        for bb in self.fn.blocks:
            self._check_block(bb, live_out)

    # ── Block-level analysis ───────────────────────────────────────────────────

    def _check_block(self, bb: BasicBlock, live_out: Dict[int, Set[int]]):
        for idx, stmt in enumerate(bb.stmts):
            self._check_stmt(stmt, bb.id, idx)
        if bb.terminator:
            self._check_terminator(bb.terminator, bb, live_out)

    def _check_stmt(self, stmt: MIRStatement, block_id: int, stmt_idx: int):
        if isinstance(stmt, StorageLive):
            self.move_state[stmt.local.index] = MoveState.Owned

        elif isinstance(stmt, StorageDead):
            # Check no loans still active on this local
            local_idx = stmt.local.index
            live_loans = self.loans_on.get(local_idx, [])
            still_live = [lid for lid in live_loans if lid in self.active_loans]
            if still_live:
                self.diag.error(
                    "E0043",
                    f"Value `{stmt.local.name or local_idx}` dropped while still borrowed",
                    hints=["ensure the borrow ends before the owner goes out of scope"]
                )
            self.move_state[local_idx] = MoveState.Moved

        elif isinstance(stmt, Assign):
            # Check RHS first (uses), then check LHS writeability
            self._check_rvalue_uses(stmt.rvalue, block_id, stmt_idx)
            self._check_place_write(stmt.place, block_id, stmt_idx)

        elif isinstance(stmt, Borrow):
            # Check the borrowed place is accessible
            local_idx = stmt.place.local.index
            state = self.move_state.get(local_idx, MoveState.Moved)

            if state == MoveState.Moved:
                self.diag.error(
                    "E0040",
                    f"Use of moved value `{stmt.place.local.name or local_idx}`",
                    hints=["value was moved earlier; you cannot borrow a moved value"]
                )
                return

            # Conflict check
            if stmt.kind == BorrowKind.Mutable:
                # Cannot mut-borrow if any active borrow exists
                existing = self.loans_on.get(local_idx, [])
                active   = [self.active_loans[lid] for lid in existing
                             if lid in self.active_loans]
                if active:
                    kinds = ["mutable" if l.kind == BorrowKind.Mutable else "shared"
                             for l in active]
                    self.diag.error(
                        "E0041",
                        f"Cannot borrow `{stmt.place}` as mutable — "
                        f"already borrowed as {kinds[0]}",
                        hints=["mutable borrows require exclusive access; "
                               "end existing borrows first"]
                    )
            else:  # Shared borrow
                existing = self.loans_on.get(local_idx, [])
                active   = [self.active_loans[lid] for lid in existing
                             if lid in self.active_loans]
                mut_borrows = [l for l in active if l.kind == BorrowKind.Mutable]
                if mut_borrows:
                    self.diag.error(
                        "E0042",
                        f"Cannot borrow `{stmt.place}` as immutable — "
                        f"already mutably borrowed",
                        hints=["end the mutable borrow before creating an immutable one"]
                    )

            # Issue new loan
            loan = Loan(
                id=self._loan_counter,
                kind=stmt.kind,
                place=stmt.place,
                issued_at=(block_id, 0),
                region=set(),
            )
            self._loan_counter += 1
            self.active_loans[loan.id] = loan
            self.loans_on.setdefault(local_idx, []).append(loan.id)

        elif isinstance(stmt, Drop):
            self._retire_loans_on(stmt.place.local.index)
            self.move_state[stmt.place.local.index] = MoveState.Moved

    def _check_terminator(self, term: Terminator, bb: BasicBlock,
                           live_out: Dict[int, Set[int]]):
        if isinstance(term, Return) and term.value:
            if isinstance(term.value, PlaceOperand):
                self._check_place_read(term.value.place)
        elif isinstance(term, Call):
            for arg in term.args:
                if isinstance(arg, PlaceOperand):
                    self._check_move(arg.place.local)
        elif isinstance(term, PromptCall):
            if isinstance(term.input, PlaceOperand):
                self._check_place_read(term.input.place)

        # Retire loans whose borrow is not live after this block
        live_after = live_out.get(bb.id, set())
        to_retire  = []
        for loan_id, loan in self.active_loans.items():
            if loan.place.local.index not in live_after:
                to_retire.append(loan_id)
        for lid in to_retire:
            del self.active_loans[lid]

    # ── Helper checks ─────────────────────────────────────────────────────────

    def _check_rvalue_uses(self, rval, block_id: int, stmt_idx: int):
        from dim_mir import BinOpRValue, UnOpRValue, UseRValue
        if isinstance(rval, UseRValue):
            if isinstance(rval.operand, PlaceOperand):
                self._check_place_read(rval.operand.place)
        elif isinstance(rval, BinOpRValue):
            for op in (rval.left, rval.right):
                if isinstance(op, PlaceOperand):
                    self._check_place_read(op.place)

    def _check_place_read(self, place: Place):
        idx   = place.local.index
        state = self.move_state.get(idx, MoveState.Moved)
        if state == MoveState.Moved:
            self.diag.error(
                "E0040",
                f"Use of moved value `{place.local.name or idx}`",
                hints=["the value was moved out of this place earlier"]
            )

    def _check_place_write(self, place: Place, block_id: int, stmt_idx: int):
        local = place.local
        # 1. Check mutability
        if local.mutability == Mutability.Not:
            state = self.move_state.get(local.index, MoveState.Moved)
            if state == MoveState.Owned:  # Only error if it was already initialised
                self.diag.error(
                    "E0044",
                    f"Cannot assign to immutable binding `{local.name or local.index}`",
                    hints=["declare with `let mut` to allow mutation"]
                )
        # 2. Check no active borrows on this place
        existing = self.loans_on.get(local.index, [])
        active   = [self.active_loans[lid] for lid in existing
                    if lid in self.active_loans]
        if active:
            kinds = ["mutable" if l.kind == BorrowKind.Mutable else "shared"
                     for l in active]
            self.diag.error(
                "E0041",
                f"Cannot assign to `{local.name or local.index}` while borrowed as {kinds[0]}",
                hints=["mutation is not allowed while a borrow is active"]
            )

    def _check_move(self, local: Local):
        idx   = local.index
        state = self.move_state.get(idx, MoveState.Moved)
        if state == MoveState.Moved:
            self.diag.error(
                "E0040",
                f"Use of moved value `{local.name or idx}`"
            )
        else:
            # Non-Copy types are moved on function call
            self.move_state[idx] = MoveState.Moved

    def _retire_loans_on(self, local_idx: int):
        for lid in self.loans_on.get(local_idx, []):
            self.active_loans.pop(lid, None)
        self.loans_on[local_idx] = []
