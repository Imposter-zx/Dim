# dim_mir_lowering.py — AST → MIR Lowering Pass
#
# Translates a type-checked, span-annotated AST into the MIR (Mid-Level IR).
# This is the pass after type checking and before code generation / borrow checking.

from __future__ import annotations
from typing import Dict, List, Optional
from dim_ast import (
    Program, Statement, Expression,
    FunctionDef, LetStmt, AssignStmt, ReturnStmt,
    IfStmt, WhileStmt, ForStmt, ExprStmt,
    Literal, Identifier, BinaryOp, UnaryOp, Call, MethodCall,
    BorrowExpr, DerefExpr, AwaitExpr, ListLiteral,
    MatchStmt, Param,
)
from dim_types import (
    Type, UNIT, I32, STR, BOOL, FutureType, UnknownType,
    FunctionType,
)
from dim_mir import (
    Local, Place, Mutability, BorrowKind,
    BasicBlock, MIRFunction, MIRModule,
    Assign, StorageLive, StorageDead, Borrow, Drop,
    Goto, Branch, Return, Call as MIRCall,
    ConstOperand, PlaceOperand, BorrowOperand,
    UseRValue, BinOpRValue, UnOpRValue,
)


class MIRLowering:
    """
    Lowers a typed AST FunctionDef into a MIRFunction.
    Typical usage:
        lowering = MIRLowering(type_ctx)
        mir_fn   = lowering.lower_function(fn_def)
    """

    def __init__(self):
        self._local_ctr = 0
        self._block_ctr = 0
        self._blocks: List[BasicBlock] = []
        self._current: Optional[BasicBlock] = None
        self._locals: Dict[str, Local] = {}   # name → Local
        self._all_locals: Dict[int, Local] = {}

    def _fresh_local(self, ty: Type, name: Optional[str] = None,
                     mut: Mutability = Mutability.Not) -> Local:
        l = Local(self._local_ctr, ty, mut, name)
        self._all_locals[l.index] = l
        self._local_ctr += 1
        return l

    def _new_block(self) -> BasicBlock:
        bb = BasicBlock(self._block_ctr)
        self._block_ctr += 1
        self._blocks.append(bb)
        return bb

    def _emit(self, stmt):
        if self._current:
            self._current.stmts.append(stmt)

    def _set_terminator(self, term):
        if self._current and self._current.terminator is None:
            self._current.terminator = term

    # ── Entry-point ───────────────────────────────────────────────────────────

    def lower_function(self, fn: FunctionDef) -> MIRFunction:
        self._local_ctr = 0
        self._block_ctr = 0
        self._blocks    = []
        self._locals    = {}
        self._all_locals = {}

        entry = self._new_block()
        self._current = entry

        # Allocate locals for parameters
        params: List[Local] = []
        for p in fn.params:
            ty  = p.type_ann if p.type_ann else UnknownType()
            mut = Mutability.Mut if p.is_mut else Mutability.Not
            loc = self._fresh_local(ty, p.name, mut)
            self._locals[p.name] = loc
            params.append(loc)
            self._emit(StorageLive(loc))

        # Lower body
        for stmt in fn.body:
            self._lower_stmt(stmt)

        # Ensure all blocks have terminators
        for bb in self._blocks:
            if bb.terminator is None:
                bb.terminator = Return(None)

        ret_ty = fn.return_type if fn.return_type else UNIT
        return MIRFunction(
            name=fn.name,
            params=params,
            return_type=ret_ty,
            locals=self._all_locals,
            blocks=self._blocks,
            is_async=fn.is_async,
        )

    # ── Statement lowering ────────────────────────────────────────────────────

    def _lower_stmt(self, stmt: Statement):
        if isinstance(stmt, LetStmt):
            ty  = stmt.value.resolved_type or UnknownType()
            mut = Mutability.Mut if stmt.is_mut else Mutability.Not
            loc = self._fresh_local(ty, stmt.name, mut)
            self._locals[stmt.name] = loc
            self._emit(StorageLive(loc))
            rval = self._lower_expr_as_rvalue(stmt.value)
            self._emit(Assign(Place(loc, ty=ty), rval))

        elif isinstance(stmt, AssignStmt):
            dest = self._lower_place(stmt.target)
            rval = self._lower_expr_as_rvalue(stmt.value)
            if dest:
                self._emit(Assign(dest, rval))

        elif isinstance(stmt, ReturnStmt):
            if stmt.value:
                op = self._lower_expr_as_operand(stmt.value)
            else:
                op = ConstOperand(None, UNIT)
            self._set_terminator(Return(op))
            # Open a new (unreachable) block for any following statements
            dead = self._new_block()
            self._current = dead

        elif isinstance(stmt, IfStmt):
            self._lower_if(stmt)

        elif isinstance(stmt, WhileStmt):
            self._lower_while(stmt)

        elif isinstance(stmt, ForStmt):
            self._lower_for(stmt)

        elif isinstance(stmt, ExprStmt):
            self._lower_expr_as_rvalue(stmt.expr)

    def _lower_if(self, stmt: IfStmt):
        cond_op  = self._lower_expr_as_operand(stmt.condition)
        then_bb  = self._new_block()
        merge_bb = self._new_block()
        else_bb  = self._new_block() if stmt.else_branch else merge_bb

        self._set_terminator(Branch(cond_op, then_bb.id, else_bb.id))

        # Then branch
        self._current = then_bb
        for s in stmt.then_branch:
            self._lower_stmt(s)
        self._set_terminator(Goto(merge_bb.id))

        # Else branch
        if stmt.else_branch:
            self._current = else_bb
            for s in stmt.else_branch:
                self._lower_stmt(s)
            self._set_terminator(Goto(merge_bb.id))

        self._current = merge_bb

    def _lower_while(self, stmt: WhileStmt):
        header_bb = self._new_block()
        body_bb   = self._new_block()
        exit_bb   = self._new_block()

        self._set_terminator(Goto(header_bb.id))

        # Header: evaluate condition
        self._current = header_bb
        cond_op = self._lower_expr_as_operand(stmt.condition)
        self._set_terminator(Branch(cond_op, body_bb.id, exit_bb.id))

        # Body
        self._current = body_bb
        for s in stmt.body:
            self._lower_stmt(s)
        self._set_terminator(Goto(header_bb.id))

        self._current = exit_bb

    def _lower_for(self, stmt: ForStmt):
        # Desugar: for x in iter → let __iter = iter; while let Some(x) = __iter.next()
        iter_ty  = stmt.iterable.resolved_type or UnknownType()
        iter_loc = self._fresh_local(iter_ty, "__iter")
        self._emit(StorageLive(iter_loc))
        iter_op  = self._lower_expr_as_operand(stmt.iterable)
        self._emit(Assign(Place(iter_loc, ty=iter_ty), UseRValue(iter_op)))

        header_bb = self._new_block()
        body_bb   = self._new_block()
        exit_bb   = self._new_block()
        self._set_terminator(Goto(header_bb.id))

        # Simplified: just loop the body (full next()/Option unwrap in Phase 2)
        self._current = header_bb
        self._set_terminator(Goto(body_bb.id))   # placeholder

        self._current = body_bb
        # Bind iterator variable
        elem_loc = self._fresh_local(UnknownType(), stmt.iterator)
        self._locals[stmt.iterator] = elem_loc
        self._emit(StorageLive(elem_loc))
        for s in stmt.body:
            self._lower_stmt(s)
        self._set_terminator(Goto(header_bb.id))   # loop back

        self._current = exit_bb

    # ── Expression lowering ───────────────────────────────────────────────────

    def _lower_expr_as_rvalue(self, expr: Expression):
        """Lower an expression into an RValue (for use on RHS of Assign)."""
        op = self._lower_expr_as_operand(expr)
        return UseRValue(op)

    def _lower_expr_as_operand(self, expr: Expression):
        """Lower an expression into an Operand."""
        ty = expr.resolved_type or UnknownType()

        if isinstance(expr, Literal):
            return ConstOperand(expr.value, ty)

        if isinstance(expr, Identifier):
            loc = self._locals.get(expr.name)
            if loc:
                return PlaceOperand(Place(loc, ty=ty))
            # Treat unknown identifiers as constants (error will be in type checker)
            dummy = self._fresh_local(ty, expr.name)
            return PlaceOperand(Place(dummy, ty=ty))

        if isinstance(expr, BinaryOp):
            left  = self._lower_expr_as_operand(expr.left)
            right = self._lower_expr_as_operand(expr.right)
            dest  = self._fresh_local(ty)
            self._emit(StorageLive(dest))
            self._emit(Assign(Place(dest, ty=ty),
                               BinOpRValue(expr.op, left, right)))
            return PlaceOperand(Place(dest, ty=ty))

        if isinstance(expr, UnaryOp):
            operand = self._lower_expr_as_operand(expr.operand)
            dest    = self._fresh_local(ty)
            self._emit(StorageLive(dest))
            self._emit(Assign(Place(dest, ty=ty),
                               UnOpRValue(expr.op, operand)))
            return PlaceOperand(Place(dest, ty=ty))

        if isinstance(expr, BorrowExpr):
            inner_place = self._lower_place(expr.expr)
            dest  = self._fresh_local(ty)
            kind  = BorrowKind.Mutable if expr.mutable else BorrowKind.Shared
            if inner_place:
                self._emit(Borrow(dest, kind, inner_place))
            return BorrowOperand(kind, Place(dest, ty=ty))

        if isinstance(expr, Call):
            callee = self._lower_expr_as_operand(expr.callee)
            args   = [self._lower_expr_as_operand(a) for a in expr.args]
            dest   = self._fresh_local(ty)
            next_bb = self._new_block()
            self._emit(StorageLive(dest))
            self._set_terminator(MIRCall(callee, args,
                                          Place(dest, ty=ty), next_bb.id))
            self._current = next_bb
            return PlaceOperand(Place(dest, ty=ty))

        # Fallback: materialise into a temp local
        dest = self._fresh_local(ty)
        self._emit(StorageLive(dest))
        return PlaceOperand(Place(dest, ty=ty))

    def _lower_place(self, expr: Expression) -> Optional[Place]:
        """Lower an expression to a writable Place (for LHS of assignment)."""
        if isinstance(expr, Identifier):
            loc = self._locals.get(expr.name)
            if loc:
                return Place(loc, ty=loc.ty)
        if isinstance(expr, DerefExpr):
            inner = self._lower_place(expr.expr)
            return inner  # simplified
        return None


# ── Module-level lowering ──────────────────────────────────────────────────────

def lower_program(prog: Program) -> MIRModule:
    """Lower all top-level functions in a Program into a MIRModule."""
    module = MIRModule(name="module")
    from dim_ast import FunctionDef
    for stmt in prog.statements:
        if isinstance(stmt, FunctionDef):
            lowering = MIRLowering()
            mir_fn   = lowering.lower_function(stmt)
            module.functions.append(mir_fn)
    return module
