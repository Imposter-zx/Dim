# dim_type_checker.py — Hindley-Milner Type Inference for Dim
from typing import Dict, Optional, List, Tuple, Set, Any
from dataclasses import dataclass, field
from dim_ast import (
    Program,
    Node,
    Statement,
    Expression,
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
    ExprStmt,
    LetStmt,
    AssignStmt,
    ReturnStmt,
    IfStmt,
    WhileStmt,
    ForStmt,
    MatchStmt,
    MatchArm,
    FunctionDef,
    StructDef,
    EnumDef,
    TraitDef,
    ImplBlock,
    PromptDef,
    Param,
    ModelCall,
    TensorExpr,
    BreakStmt,
    ContinueStmt,
)
from dim_types import (
    Type,
    TypeVar,
    FunctionType,
    TensorType,
    PromptType,
    GenericType,
    StructType,
    EnumType,
    FutureType,
    UnknownType,
    RefType,
    PrimType,
    I32,
    I64,
    F32,
    F64,
    BOOL,
    STR,
    UNIT,
    BUILTIN_TYPES,
    resolve_builtin,
    numeric_promotion,
)
from dim_diagnostic import DiagnosticBag, Span


@dataclass
class Symbol:
    name: str
    ty: Type
    is_mut: bool = False
    span: Any = None


class TypeEnv:
    def __init__(self):
        self._scopes: List[Dict[str, Symbol]] = [{}]
        self._substitutions: Dict[str, Type] = {}

    def push(self):
        self._scopes.append({})

    def pop(self):
        self._scopes.pop()

    def define(self, sym: Symbol):
        self._scopes[-1][sym.name] = sym

    def lookup(self, name: str) -> Optional[Symbol]:
        for s in reversed(self._scopes):
            if name in s:
                return s[name]
        return None

    def substitute(self, ty: Type) -> Type:
        if isinstance(ty, TypeVar):
            return self._substitutions.get(ty.name, ty)
        return ty

    def unify(self, t1: Type, t2: Type, span: Any = None) -> bool:
        t1 = self.substitute(t1)
        t2 = self.substitute(t2)

        if t1 == t2:
            return True
        if isinstance(t1, UnknownType):
            return True
        if isinstance(t2, UnknownType):
            return True
        if isinstance(t1, TypeVar):
            if self._occurs_in(t1.name, t2):
                self.diag.error("E0031", "Recursive type detected", span)
                return False
            self._substitutions[t1.name] = t2
            return True
        if isinstance(t2, TypeVar):
            return self.unify(t2, t1, span)
        if isinstance(t1, PrimType) and isinstance(t2, PrimType):
            return t1 == t2
        if isinstance(t1, FunctionType) and isinstance(t2, FunctionType):
            if len(t1.params) != len(t2.params):
                self.diag.error("E0032", "Argument count mismatch", span)
                return False
            for p, q in zip(t1.params, t2.params):
                self.unify(p, q, span)
            self.unify(t1.return_type, t2.return_type, span)
            return True
        if isinstance(t1, TensorType) and isinstance(t2, TensorType):
            self.unify(t1.dtype, t2.dtype, span)
            return True
        if isinstance(t1, RefType) and isinstance(t2, RefType):
            if t1.mutable != t2.mutable:
                self.diag.error("E0030", "Reference mutability mismatch", span)
            return self.unify(t1.inner, t2.inner, span)
        if isinstance(t1, PromptType) and isinstance(t2, PromptType):
            return True
        return False

    def _occurs_in(self, var: str, ty: Type) -> bool:
        if isinstance(ty, TypeVar):
            return ty.name == var
        if isinstance(ty, FunctionType):
            return any(self._occurs_in(var, p) for p in ty.params) or self._occurs_in(
                var, ty.return_type
            )
        if isinstance(ty, TensorType):
            return self._occurs_in(var, ty.dtype)
        return False


class TypeChecker:
    def __init__(self, source: str = "", filename: str = ""):
        self.env = TypeEnv()
        self.diag = DiagnosticBag(source, filename)
        self._current_fn_return: Optional[Type] = None
        self.current_capabilities: Set[str] = set()
        self._var_counter = 0
        for n, ty in BUILTIN_TYPES.items():
            self.env.define(Symbol(n, ty))
        self.env.define(Symbol("model", GenericType("Model", [])))
        self.env.define(Symbol("print", FunctionType([GenericType("T", [])], UNIT)))
        self._structs: Dict[str, StructDef] = {}
        self._enums: Dict[str, EnumDef] = {}

    def fresh_var(self) -> TypeVar:
        self._var_counter += 1
        return TypeVar("t" + str(self._var_counter))

    def resolve_type(self, ty: Optional[Type]) -> Type:
        if ty is None:
            return self.fresh_var()
        return ty

    def check_program(self, prog: Program):
        for s in prog.statements:
            if isinstance(s, (FunctionDef, PromptDef, StructDef, EnumDef)):
                self._hoist(s)
        for s in prog.statements:
            self.check_stmt(s)

    def _hoist(self, stmt):
        if isinstance(stmt, FunctionDef):
            ps = [self.resolve_type(p.type_ann) for p in stmt.params]
            rt = self.resolve_type(stmt.return_type)
            ty = FunctionType(ps, rt, stmt.is_async, capabilities=stmt.capabilities)
            self.env.define(Symbol(stmt.name, ty, span=stmt.span))
            stmt.resolved_fn_type = ty
        elif isinstance(stmt, PromptDef):
            in_ty = self.resolve_type(stmt.input_type)
            out_ty = self.resolve_type(stmt.output_type)
            ty = PromptType(stmt.name, in_ty, out_ty, stmt.deterministic)
            self.env.define(Symbol(stmt.name, ty, span=stmt.span))
        elif isinstance(stmt, StructDef):
            self._structs[stmt.name] = stmt
            fields = {fname: ftype for fname, ftype, _ in stmt.fields}
            self.env.define(Symbol(stmt.name, StructType(stmt.name, fields)))
        elif isinstance(stmt, EnumDef):
            self._enums[stmt.name] = stmt
            variants = {vname: vtypes for vname, vtypes in stmt.variants}
            self.env.define(Symbol(stmt.name, EnumType(stmt.name, variants)))

    def check_stmt(self, stmt: Statement):
        from dim_ast import (
            LetStmt,
            ReturnStmt,
            ExprStmt,
            IfStmt,
            WhileStmt,
            ForStmt,
            AssignStmt,
            BreakStmt,
            ContinueStmt,
            MatchStmt,
        )

        if isinstance(stmt, FunctionDef):
            self.env.push()
            old_ret = self._current_fn_return
            old_caps = self.current_capabilities
            self._current_fn_return = stmt.resolved_fn_type.return_type
            self.current_capabilities = set(stmt.capabilities)
            for p, ty in zip(stmt.params, stmt.resolved_fn_type.params):
                self.env.define(Symbol(p.name, ty, p.is_mut, p.span))
            for s in stmt.body:
                self.check_stmt(s)
            self._current_fn_return = old_ret
            self.current_capabilities = old_caps
            self.env.pop()

        elif isinstance(stmt, LetStmt):
            ty = self.infer(stmt.value)
            self.env.define(Symbol(stmt.name, ty, stmt.is_mut, stmt.span))

        elif isinstance(stmt, AssignStmt):
            from dim_ast import Identifier

            val_ty = self.infer(stmt.value)
            tgt_ty = (
                self.infer(stmt.target) if isinstance(stmt.target, Expression) else None
            )
            if isinstance(stmt.target, Identifier):
                sym = self.env.lookup(stmt.target.name)
                if sym and not sym.is_mut:
                    self.diag.error(
                        "E0044",
                        f"Cannot assign to immutable binding `{stmt.target.name}`",
                        stmt.target.span,
                        hints=["declare with `let mut` to allow mutation"],
                    )
            if tgt_ty:
                self.env.unify(tgt_ty, val_ty, stmt.span)

        elif isinstance(stmt, ReturnStmt):
            if stmt.value:
                ret_ty = self.infer(stmt.value)
                if self._current_fn_return:
                    self.env.unify(ret_ty, self._current_fn_return, stmt.span)

        elif isinstance(stmt, ExprStmt):
            self.infer(stmt.expr)

        elif isinstance(stmt, AssignStmt):
            val_ty = self.infer(stmt.value)
            tgt_ty = (
                self.infer(stmt.target) if isinstance(stmt.target, Expression) else None
            )
            if tgt_ty and val_ty != UnknownType():
                if not self.env.unify(tgt_ty, val_ty, stmt.span):
                    self.diag.error(
                        "E0030", f"Cannot assign `{val_ty}` to `{tgt_ty}`", stmt.span
                    )

        elif isinstance(stmt, IfStmt):
            cond_ty = self.infer(stmt.condition)
            self.env.unify(cond_ty, BOOL, stmt.condition.span)
            for s in stmt.then_branch:
                self.check_stmt(s)
            for _, elif_body in stmt.elif_branches:
                for s in elif_body:
                    self.check_stmt(s)
            if stmt.else_branch:
                for s in stmt.else_branch:
                    self.check_stmt(s)

        elif isinstance(stmt, WhileStmt):
            cond_ty = self.infer(stmt.condition)
            self.env.unify(cond_ty, BOOL, stmt.condition.span)
            for s in stmt.body:
                self.check_stmt(s)

        elif isinstance(stmt, ForStmt):
            iter_ty = self.infer(stmt.iterable)
            self.env.define(Symbol(stmt.iterator, I32, span=stmt.span))
            for s in stmt.body:
                self.check_stmt(s)

        elif isinstance(stmt, MatchStmt):
            expr_ty = self.infer(stmt.expr)
            for arm in stmt.arms:
                for s in arm.body:
                    self.check_stmt(s)

    def infer(self, expr: Expression) -> Type:
        ty = self._infer_inner(expr)
        ty = self.env.substitute(ty)
        expr.resolved_type = ty
        return ty

    def _infer_inner(self, expr: Expression) -> Type:
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
            TensorExpr,
            ModelCall,
        )

        if isinstance(expr, Literal):
            if isinstance(expr.value, bool):
                return BOOL
            if isinstance(expr.value, int):
                return I32
            if isinstance(expr.value, float):
                return F32
            if isinstance(expr.value, str):
                return STR
            if expr.value is None:
                return UNIT
            return UnknownType()

        if isinstance(expr, Identifier):
            sym = self.env.lookup(expr.name)
            if sym:
                return sym.ty
            self.diag.error("E0020", f"Undefined variable `{expr.name}`", expr.span)
            return UnknownType()

        if isinstance(expr, BinaryOp):
            lt = self.infer(expr.left)
            rt = self.infer(expr.right)
            result = numeric_promotion(lt, rt)
            if result is None:
                self.diag.error(
                    "E0030", f"Cannot apply `{expr.op}` to `{lt}` and `{rt}`", expr.span
                )
                return UnknownType()
            if expr.op in ("==", "!=", "<", ">", "<=", ">="):
                return BOOL
            return result

        if isinstance(expr, UnaryOp):
            op_ty = self.infer(expr.operand)
            if expr.op == "not":
                self.env.unify(op_ty, BOOL, expr.span)
                return BOOL
            return op_ty

        if isinstance(expr, Call):
            return self._infer_call(expr)

        if isinstance(expr, MethodCall):
            return self._infer_method_call(expr)

        if isinstance(expr, BorrowExpr):
            inner_ty = self.infer(expr.expr)
            return RefType(inner_ty, expr.mutable)

        if isinstance(expr, DerefExpr):
            inner_ty = self.infer(expr.expr)
            if isinstance(inner_ty, RefType):
                return inner_ty.inner
            self.diag.error("E0030", f"Cannot dereference type `{inner_ty}`", expr.span)
            return UnknownType()

        if isinstance(expr, AwaitExpr):
            inner_ty = self.infer(expr.expr)
            if isinstance(inner_ty, FutureType):
                return inner_ty.inner
            if isinstance(inner_ty, GenericType):
                return UnknownType()
            self.diag.error("E0030", f"Cannot await type `{inner_ty}`", expr.span)
            return UnknownType()

        if isinstance(expr, ListLiteral):
            if expr.elements:
                elem_ty = self.infer(expr.elements[0])
                return TensorType(elem_ty, [len(expr.elements)])
            return TensorType(UnknownType(), [0])

        if isinstance(expr, TensorExpr):
            dtype = resolve_builtin(expr.dtype) or self.fresh_var()
            return TensorType(dtype, expr.shape)

        if isinstance(expr, ModelCall):
            input_ty = self.infer(expr.input)
            return GenericType("Model", [input_ty])

        if isinstance(expr, TupleLiteral):
            if expr.elements:
                elem_tys = [self.infer(e) for e in expr.elements]
                return GenericType("tuple", elem_tys)
            return GenericType("tuple", [])

        if isinstance(expr, MemberAccess):
            obj_ty = self.infer(expr.expr)
            if isinstance(obj_ty, StructType):
                if expr.member in obj_ty.fields:
                    return obj_ty.fields[expr.member]
                self.diag.error(
                    "E0020",
                    f"Struct `{obj_ty.name}` has no field `{expr.member}`",
                    expr.span,
                )
            return self.fresh_var()

        if isinstance(expr, IndexAccess):
            self.infer(expr.expr)
            self.infer(expr.index)
            return self.fresh_var()

        if isinstance(expr, ClosureExpr):
            param_tys = [self.resolve_type(p.type_ann) for p in expr.params]
            ret_ty = self.fresh_var()
            self.env.push()
            for p, ty in zip(expr.params, param_tys):
                self.env.define(Symbol(p.name, ty, p.is_mut, p.span))
            for s in expr.body:
                self.check_stmt(s)
            self.env.pop()
            return FunctionType(param_tys, ret_ty)

        return UnknownType()

    def _infer_call(self, expr: Call) -> Type:
        fn_ty = self.infer(expr.callee)

        if isinstance(fn_ty, FunctionType):
            for cap in fn_ty.capabilities:
                if cap not in self.current_capabilities:
                    self.diag.error(
                        "E0061", f"Missing capability `{cap}` for this call", expr.span
                    )

            if len(fn_ty.params) != len(expr.args):
                self.diag.error(
                    "E0032",
                    f"Expected {len(fn_ty.params)} arguments, got {len(expr.args)}",
                    expr.span,
                )

            for arg, param_ty in zip(expr.args, fn_ty.params):
                arg_ty = self.infer(arg)
                self.env.unify(param_ty, arg_ty, expr.span)

            return fn_ty.return_type

        if isinstance(fn_ty, TypeVar):
            ret_ty = self.fresh_var()
            param_tys = [self.infer(a) for a in expr.args]
            self.env.unify(fn_ty, FunctionType(param_tys, ret_ty), expr.span)
            return ret_ty

        self.diag.error("E0021", f"`{expr.callee}` is not callable", expr.span)
        return UnknownType()

    def _infer_method_call(self, expr: MethodCall) -> Type:
        recv_ty = self.infer(expr.receiver)

        if isinstance(recv_ty, RefType) and expr.method == "deref":
            return recv_ty.inner

        if isinstance(recv_ty, GenericType) and recv_ty.name == "Model":
            if expr.method == "execute":
                if expr.args:
                    p_ty = self.infer(expr.args[0])
                    if isinstance(p_ty, PromptType):
                        return FutureType(p_ty.output_type)
                    return FutureType(self.fresh_var())
                return FutureType(self.fresh_var())

        return UnknownType()
