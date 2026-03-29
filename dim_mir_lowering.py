# dim_mir_lowering.py — AST → MIR Lowering Pass
from typing import Dict, List, Optional, Any, Tuple
from dim_ast import (
    Program,
    Statement,
    Expression,
    FunctionDef,
    LetStmt,
    AssignStmt,
    ReturnStmt,
    IfStmt,
    WhileStmt,
    ForStmt,
    ExprStmt,
    MatchStmt,
    Literal,
    Identifier,
    BinaryOp,
    UnaryOp,
    Call,
    MethodCall,
    BorrowExpr,
    DerefExpr,
    AwaitExpr,
    ListLiteral,
    Param,
    TensorExpr,
    ModelCall,
)
from dim_types import (
    Type,
    UNIT,
    I32,
    STR,
    BOOL,
    FutureType,
    UnknownType,
    FunctionType,
    PromptType,
    GenericType,
    TensorType,
)
from dim_mir import (
    Local,
    Place,
    BasicBlock,
    MIRFunction,
    MIRModule,
    Assign,
    StorageLive,
    StorageDead,
    Goto,
    Branch,
    Return,
    Call as MIRCall,
    PromptCall,
    ConstOperand,
    PlaceOperand,
    UseRValue,
    BinOpRValue,
    UnOpRValue,
    TensorRValue,
    Borrow,
    Drop,
    Mutability,
    BorrowKind,
)


class LoweringPass:
    def __init__(self):
        self.functions = []
        self.current_fn: Optional[MIRFunction] = None
        self.local_map: Dict[str, Local] = {}
        self.block_counter = 0
        self.temp_counter = 0
        self._loop_stack: List[Dict] = []

    def lower(self, prog):
        module = MIRModule("main")
        for s in prog.statements:
            if isinstance(s, FunctionDef):
                self.functions.append(self.lower_function(s))
        module.functions = self.functions
        return module

    def lower_function(self, fn: FunctionDef) -> MIRFunction:
        from dim_types import TypeVar

        ret_ty = fn.resolved_fn_type.return_type
        if isinstance(ret_ty, TypeVar):
            ret_ty = UNIT
        mir_fn = MIRFunction(fn.name, [], ret_ty)
        self.current_fn = mir_fn
        self.local_map = {}
        self.block_counter = 0
        self.temp_counter = 0
        self._loop_stack = []

        bb0 = BasicBlock(0)
        mir_fn.blocks.append(bb0)

        params: List[Local] = []
        for p in fn.params:
            ty = p.type_ann or I32
            loc = Local(
                len(mir_fn.locals),
                p.name,
                ty,
                Mutability.Mut if p.is_mut else Mutability.Not,
            )
            mir_fn.locals.append(loc)
            mir_fn.locals_map[loc.index] = loc
            self.local_map[p.name] = loc
            params.append(loc)
            bb0.stmts.append(StorageLive(loc))

        mir_fn.params = params

        current_bb = bb0
        pending_cont: Optional[BasicBlock] = None
        for s in fn.body:
            if pending_cont is not None:
                current_bb = pending_cont
                pending_cont = None
            elif current_bb.terminator is not None:
                current_bb = self._new_block()
                mir_fn.blocks.append(current_bb)
            pending_cont = self.lower_stmt(s, current_bb, mir_fn)
            if pending_cont is not None and pending_cont not in mir_fn.blocks:
                mir_fn.blocks.append(pending_cont)

        if current_bb.terminator is None:
            current_bb.terminator = Return(ConstOperand(UNIT, None))
        if pending_cont is not None and pending_cont not in mir_fn.blocks:
            mir_fn.blocks.append(pending_cont)
        elif current_bb not in mir_fn.blocks:
            mir_fn.blocks.append(current_bb)

        return mir_fn

    def _new_block(self) -> BasicBlock:
        bb = BasicBlock(self.block_counter)
        self.block_counter += 1
        return bb

    def _new_temp(self, ty: Type) -> Local:
        mir_fn = self.current_fn
        loc = Local(len(mir_fn.locals), f"_t{self.temp_counter}", ty)
        self.temp_counter += 1
        mir_fn.locals.append(loc)
        mir_fn.locals_map[loc.index] = loc
        return loc

    def lower_stmt(
        self, stmt: Statement, bb: BasicBlock, mir_fn: MIRFunction
    ) -> Tuple[Optional[BasicBlock], bool]:
        from dim_ast import (
            Identifier,
            AssignStmt,
            IfStmt,
            WhileStmt,
            ForStmt,
            ReturnStmt,
            BreakStmt,
            ContinueStmt,
            MatchStmt,
            ExprStmt,
            TryStmt,
            ThrowStmt,
        )

        cont: Optional[BasicBlock] = None

        if isinstance(stmt, LetStmt):
            ty = (
                stmt.value.resolved_type
                if hasattr(stmt.value, "resolved_type")
                else I32
            )
            loc = Local(
                len(mir_fn.locals),
                stmt.name,
                ty,
                Mutability.Mut if stmt.is_mut else Mutability.Not,
            )
            mir_fn.locals.append(loc)
            mir_fn.locals_map[loc.index] = loc
            self.local_map[stmt.name] = loc
            bb.stmts.append(StorageLive(loc))
            if isinstance(stmt.value, Call):
                args_ops = []
                for a in stmt.value.args:
                    a_mir, _ = self.lower_expr(a, bb)
                    args_ops.append(self._to_operand(a_mir, bb))
                fn_name = (
                    stmt.value.callee.name
                    if isinstance(stmt.value.callee, Identifier)
                    else "__dim_unknown_fn"
                )
                next_bb = self._new_block()
                bb.terminator = MIRCall(fn_name, args_ops, Place(loc), next_bb)
                cont = next_bb
            else:
                val, _ = self.lower_expr(stmt.value, bb)
                rval = val if val else UseRValue(ConstOperand(ty, None))
                bb.stmts.append(Assign(Place(loc), rval))

        elif isinstance(stmt, AssignStmt):
            if stmt.op == "=":
                val, _ = self.lower_expr(stmt.value, bb)
                if isinstance(stmt.target, Identifier):
                    loc = self.local_map.get(stmt.target.name)
                    if loc:
                        rval = val if val else UseRValue(ConstOperand(loc.ty, None))
                        bb.stmts.append(Assign(Place(loc), rval))
            else:
                target_val, _ = self.lower_expr(stmt.target, bb)
                rhs_val, _ = self.lower_expr(stmt.value, bb)
                op_map = {"+=": "+", "-=": "-", "*=": "*", "/=": "/", "%=": "%"}
                bin_op = op_map.get(stmt.op, stmt.op)
                if isinstance(stmt.target, Identifier):
                    loc = self.local_map.get(stmt.target.name)
                    if loc:
                        lhs = target_val if target_val else Place(loc)
                        rhs = (
                            rhs_val
                            if rhs_val
                            else UseRValue(ConstOperand(loc.ty, None))
                        )
                        rval = BinOpRValue(lhs, bin_op, rhs)
                        bb.stmts.append(Assign(Place(loc), rval))

        elif isinstance(stmt, ReturnStmt):
            val, _ = self.lower_expr(stmt.value, bb) if stmt.value else (None, None)
            bb.terminator = Return(self._to_operand(val, bb))

        elif isinstance(stmt, TryStmt):
            for s in stmt.body:
                self.lower_stmt(s, bb, mir_fn)
            if stmt.finally_body:
                for s in stmt.finally_body:
                    self.lower_stmt(s, bb, mir_fn)

        elif isinstance(stmt, ThrowStmt):
            pass

        elif isinstance(stmt, ExprStmt):
            _, next_bb = self.lower_expr(stmt.expr, bb)
            if next_bb is not None:
                cont = next_bb

        elif isinstance(stmt, IfStmt):
            cond, _ = self.lower_expr(stmt.condition, bb)
            cond_op = self._to_operand(cond, bb)

            then_bb = self._new_block()
            else_bb = (
                self._new_block() if stmt.else_branch or stmt.elif_branches else None
            )
            merge_bb = self._new_block()

            bb.terminator = Branch(
                cond_op,
                true_target=then_bb.id,
                false_target=else_bb.id if else_bb else merge_bb.id,
            )

            mir_fn.blocks.append(then_bb)
            for s in stmt.then_branch:
                self.lower_stmt(s, then_bb, mir_fn)
            if then_bb.terminator is None:
                then_bb.terminator = Goto(merge_bb.id)

            if else_bb:
                mir_fn.blocks.append(else_bb)
                for s in stmt.else_branch or []:
                    self.lower_stmt(s, else_bb, mir_fn)
                if else_bb.terminator is None:
                    else_bb.terminator = Goto(merge_bb.id)

            mir_fn.blocks.append(merge_bb)

        elif isinstance(stmt, WhileStmt):
            cond_bb = self._new_block()
            body_bb = self._new_block()
            merge_bb = self._new_block()

            self._loop_stack.append({"merge": merge_bb, "cond": cond_bb})

            bb.terminator = Goto(cond_bb.id)

            cond, _ = self.lower_expr(stmt.condition, cond_bb)
            cond_bb.terminator = Branch(
                self._to_operand(cond, cond_bb),
                true_target=body_bb.id,
                false_target=merge_bb.id,
            )

            mir_fn.blocks.append(cond_bb)
            mir_fn.blocks.append(body_bb)
            for s in stmt.body:
                self.lower_stmt(s, body_bb, mir_fn)
            if body_bb.terminator is None:
                body_bb.terminator = Goto(cond_bb.id)

            mir_fn.blocks.append(merge_bb)
            self._loop_stack.pop()

        elif isinstance(stmt, ForStmt):
            iter_val, _ = self.lower_expr(stmt.iterable, bb)
            iter_loc = self._new_temp(I32)
            bb.stmts.append(StorageLive(iter_loc))
            bb.stmts.append(Assign(Place(iter_loc), iter_val))

            body_bb = self._new_block()
            merge_bb = self._new_block()

            self._loop_stack.append({"merge": merge_bb, "cond": body_bb})

            self.local_map[stmt.iterator] = iter_loc
            mir_fn.locals.append(iter_loc)
            mir_fn.locals_map[iter_loc.index] = iter_loc

            bb.terminator = Goto(body_bb.id)
            mir_fn.blocks.append(body_bb)

            for s in stmt.body:
                self.lower_stmt(s, body_bb, mir_fn)
            if body_bb.terminator is None:
                body_bb.terminator = Goto(merge_bb.id)

            mir_fn.blocks.append(merge_bb)
            self._loop_stack.pop()

        elif isinstance(stmt, BreakStmt):
            if self._loop_stack:
                merge_bb = self._loop_stack[-1]["merge"]
                bb.terminator = Goto(merge_bb.id)
            else:
                bb.terminator = Goto(0)

        elif isinstance(stmt, ContinueStmt):
            if self._loop_stack:
                cond_bb = self._loop_stack[-1]["cond"]
                bb.terminator = Goto(cond_bb.id)
            else:
                bb.terminator = Goto(0)

        elif isinstance(stmt, MatchStmt):
            expr_val, _ = self.lower_expr(stmt.expr, bb)
            merge_bb = self._new_block()
            prev_bb = bb
            for i, arm in enumerate(stmt.arms):
                arm_bb = self._new_block()
                mir_fn.blocks.append(arm_bb)
                for s in arm.body:
                    self.lower_stmt(s, arm_bb, mir_fn)
                if arm_bb.terminator is None:
                    arm_bb.terminator = Goto(merge_bb.id)
                if i == 0:
                    prev_bb.terminator = Goto(arm_bb.id)
                else:
                    prev_bb.terminator = Goto(arm_bb.id)
                prev_bb = arm_bb
            mir_fn.blocks.append(merge_bb)

        return cont

    def lower_expr(
        self, expr: Expression, bb: BasicBlock
    ) -> Tuple[Any, Optional[BasicBlock]]:
        from dim_ast import (
            Literal,
            Identifier,
            BinaryOp,
            UnaryOp,
            Call,
            MethodCall,
            BorrowExpr,
            DerefExpr,
            AwaitExpr,
            ListLiteral,
            TupleLiteral,
            MemberAccess,
            IndexAccess,
            ClosureExpr,
            StructConstruct,
            TensorExpr,
        )

        if isinstance(expr, Literal):
            return UseRValue(ConstOperand(expr.resolved_type or I32, expr.value)), None

        if isinstance(expr, Identifier):
            loc = self.local_map.get(expr.name)
            if loc:
                return UseRValue(PlaceOperand(Place(loc))), None
            return UseRValue(ConstOperand(UnknownType(), None)), None

        if isinstance(expr, BinaryOp):
            l, _ = self.lower_expr(expr.left, bb)
            r, _ = self.lower_expr(expr.right, bb)
            lop = self._to_operand(l, bb)
            rop = self._to_operand(r, bb)
            lop.ty = (
                expr.left.resolved_type if hasattr(expr.left, "resolved_type") else I32
            )
            rop.ty = (
                expr.right.resolved_type
                if hasattr(expr.right, "resolved_type")
                else I32
            )
            return BinOpRValue(lop, expr.op, rop), None

        if isinstance(expr, UnaryOp):
            operand, _ = self.lower_expr(expr.operand, bb)
            return UnOpRValue(expr.op, self._to_operand(operand, bb)), None

        if isinstance(expr, Call):
            return Place(self._new_temp(expr.resolved_type or I32)), None

        if isinstance(expr, MethodCall) and expr.method == "execute":
            val, next_bb = self.lower_model_execute(expr, bb)
            return val, next_bb

        if isinstance(expr, BorrowExpr):
            loc = (
                self.local_map.get(
                    expr.expr.name if isinstance(expr.expr, Identifier) else None
                )
                if isinstance(expr.expr, Identifier)
                else None
            )
            if loc:
                bk = BorrowKind.Mutable if expr.mutable else BorrowKind.Shared
                temp = self._new_temp(loc.ty)
                bb.stmts.append(StorageLive(temp))
                bb.stmts.append(Borrow(Place(temp), bk, Place(loc)))
                return UseRValue(PlaceOperand(Place(temp))), None

        if isinstance(expr, TensorExpr):
            return TensorRValue(
                expr.resolved_type.dtype
                if hasattr(expr.resolved_type, "dtype")
                else F32,
                expr.shape,
            ), None

        if isinstance(expr, ListLiteral):
            return TensorRValue(
                expr.resolved_type.dtype
                if hasattr(expr.resolved_type, "dtype")
                else F32,
                [len(expr.elements)],
            ), None

        if isinstance(expr, TupleLiteral):
            return UseRValue(ConstOperand(I32, 0)), None

        if isinstance(expr, MemberAccess):
            return UseRValue(ConstOperand(I32, 0)), None

        if isinstance(expr, IndexAccess):
            return UseRValue(ConstOperand(I32, 0)), None

        if isinstance(expr, ClosureExpr):
            return UseRValue(ConstOperand(I32, 0)), None

        if isinstance(expr, StructConstruct):
            return UseRValue(ConstOperand(I32, 0)), None

        return UseRValue(ConstOperand(I32, 0)), None

    def lower_model_execute(self, expr, bb):
        prompt_arg = expr.args[0] if expr.args else None
        if prompt_arg:
            input_mir, _ = self.lower_expr(prompt_arg, bb)
            temp_loc = self._new_temp(expr.resolved_type or UnknownType())
            bb.stmts.append(StorageLive(temp_loc))
            next_bb = self._new_block()
            bb.terminator = PromptCall(
                prompt_ref=getattr(prompt_arg.resolved_type, "name", "Prompt"),
                input=self._to_operand(input_mir, bb),
                dest=Place(temp_loc),
                next_block=next_bb,
            )
            return UseRValue(PlaceOperand(Place(temp_loc))), next_bb
        return UseRValue(ConstOperand(UnknownType(), None)), None

    def _to_operand(self, rval, bb) -> Any:
        if rval is None:
            return ConstOperand(I32, None)
        if isinstance(rval, Place):
            return PlaceOperand(rval)
        if isinstance(rval, UseRValue):
            return rval.operand
        if isinstance(rval, BinOpRValue):
            temp = self._new_temp(I32)
            bb.stmts.append(Assign(Place(temp), rval))
            return PlaceOperand(Place(temp))
        if isinstance(rval, TensorRValue):
            temp = self._new_temp(I32)
            bb.stmts.append(Assign(Place(temp), rval))
            return PlaceOperand(Place(temp))
        return ConstOperand(I32, None)


def lower_program(prog: Program) -> MIRModule:
    return LoweringPass().lower(prog)
