# dim_type_checker.py — Hindley-Milner Type Inference for Dim
#
# Bidirectional type checker operating on the span-annotated AST.
# Produces a typed AST (resolved_type filled on every Expression node)
# and emits errors through DiagnosticBag.

from __future__ import annotations
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field

from dim_ast import (
    Program, Node, Statement, Expression,
    Literal, Identifier, BinaryOp, UnaryOp, Call, MethodCall,
    MemberAccess, IndexAccess, BorrowExpr, DerefExpr, AwaitExpr,
    ListLiteral, ClosureExpr, IfExpr,
    LetStmt, AssignStmt, ReturnStmt, IfStmt, WhileStmt, ForStmt,
    MatchStmt, ExprStmt,
    FunctionDef, StructDef, EnumDef, TraitDef, ImplBlock,
    PromptDef, Param, Program,
    ModelCall,
)
from dim_types import (
    Type, PrimType, PrimKind, TypeVar, RefType, FunctionType,
    GenericType, StructType, EnumType, TraitType, TensorType,
    PromptType, FutureType, ResultType, OptionType, UnknownType,
    I32, I64, F32, F64, BOOL, STR, UNIT, NEVER,
    BUILTIN_TYPES, resolve_builtin, numeric_promotion,
)
from dim_diagnostic import DiagnosticBag
from dim_token import Span


@dataclass
class Symbol:
    name: str
    ty:   Type
    is_mut:  bool = False
    span:    Optional[Span] = None


class TypeEnv:
    """Scoped symbol table mapping names to typed symbols."""

    def __init__(self):
        self._scopes: List[Dict[str, Symbol]] = [{}]

    def push(self):
        self._scopes.append({})

    def pop(self):
        self._scopes.pop()

    def define(self, sym: Symbol):
        self._scopes[-1][sym.name] = sym

    def lookup(self, name: str) -> Optional[Symbol]:
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return None

    def redefine(self, name: str, ty: Type):
        """Update type of an already-defined symbol (for inference)."""
        for scope in reversed(self._scopes):
            if name in scope:
                scope[name].ty = ty
                return


class TypeChecker:
    """
    Bidirectional Hindley-Milner type checker.
    - infer(expr) → Type  : synthesise type from expression
    - check(expr, expected): verify expression matches an expected type
    """

    def __init__(self, source: str = "", filename: str = "<unknown>"):
        self.env  = TypeEnv()
        self.diag = DiagnosticBag(source, filename)
        self._current_fn_return: Optional[Type] = None
        self._var_counter = 0

        # Register built-in types in the top-level env
        for name, ty in BUILTIN_TYPES.items():
            self.env.define(Symbol(name=name, ty=ty))

    # ── Fresh type variable ───────────────────────────────────────────────────

    def fresh_var(self, hint: str = "t") -> TypeVar:
        self._var_counter += 1
        return TypeVar(f"{hint}{self._var_counter}")

    # ── Resolve type annotations from name strings ────────────────────────────

    def resolve_type(self, ty: Optional[Type]) -> Type:
        if ty is None:
            return self.fresh_var()
        return ty

    def _resolve_type_name(self, name: str) -> Type:
        builtin = resolve_builtin(name)
        if builtin:
            return builtin
        sym = self.env.lookup(name)
        if sym and isinstance(sym.ty, (StructType, EnumType, TraitType)):
            return sym.ty
        return UnknownType()

    # ── Top-level ─────────────────────────────────────────────────────────────

    def check_program(self, prog: Program):
        """Two-pass: first collect all top-level definitions, then check bodies."""
        # Pass 1: hoist function + struct + enum + trait names
        for stmt in prog.statements:
            self._hoist(stmt)
        # Pass 2: full analysis
        for stmt in prog.statements:
            self.check_stmt(stmt)

    def _hoist(self, stmt):
        if isinstance(stmt, FunctionDef):
            params = [self.resolve_type(p.type_ann) for p in stmt.params]
            ret    = self.resolve_type(stmt.return_type)
            fn_ty  = FunctionType(params, ret, stmt.is_async)
            self.env.define(Symbol(stmt.name, fn_ty, span=stmt.span))
            stmt.resolved_fn_type = fn_ty
        elif isinstance(stmt, StructDef):
            fields = {n: self.resolve_type(t) for n, t, *_ in stmt.fields}
            ty = StructType(stmt.name, fields, stmt.generics)
            self.env.define(Symbol(stmt.name, ty, span=stmt.span))
        elif isinstance(stmt, EnumDef):
            variants = {n: ([self.resolve_type(t) for t in ts] if ts else None)
                        for n, *ts in stmt.variants}
            ty = EnumType(stmt.name, variants, stmt.generics)
            self.env.define(Symbol(stmt.name, ty, span=stmt.span))
        elif isinstance(stmt, TraitDef):
            ty = TraitType(stmt.name, {})
            self.env.define(Symbol(stmt.name, ty, span=stmt.span))
        elif isinstance(stmt, PromptDef):
            in_ty  = self.resolve_type(stmt.input_type)
            out_ty = self.resolve_type(stmt.output_type)
            ty = PromptType(in_ty, out_ty, stmt.deterministic)
            self.env.define(Symbol(stmt.name, ty, span=stmt.span))

    # ── Statement checking ─────────────────────────────────────────────────────

    def check_stmt(self, stmt: Statement):
        if isinstance(stmt, FunctionDef):
            self._check_function(stmt)
        elif isinstance(stmt, LetStmt):
            self._check_let(stmt)
        elif isinstance(stmt, AssignStmt):
            self._check_assign(stmt)
        elif isinstance(stmt, ReturnStmt):
            self._check_return(stmt)
        elif isinstance(stmt, IfStmt):
            self._check_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self.env.push()
            self.infer(stmt.condition)
            self._check_block(stmt.body)
            self.env.pop()
        elif isinstance(stmt, ForStmt):
            iter_ty = self.infer(stmt.iterable)
            elem_ty = self._iter_element_type(iter_ty)
            self.env.push()
            self.env.define(Symbol(stmt.iterator, elem_ty, stmt.span))
            self._check_block(stmt.body)
            self.env.pop()
        elif isinstance(stmt, MatchStmt):
            self.infer(stmt.expression)
            for arm in stmt.arms:
                self.env.push()
                if arm.guard:
                    self.infer(arm.guard)
                self._check_block(arm.body)
                self.env.pop()
        elif isinstance(stmt, ExprStmt):
            self.infer(stmt.expr)
        elif isinstance(stmt, (StructDef, EnumDef, TraitDef, PromptDef)):
            pass  # hoisted in pass 1

    def _check_block(self, stmts: List[Statement]):
        for s in stmts:
            self.check_stmt(s)

    def _check_function(self, fn: FunctionDef):
        self.env.push()
        param_types = []
        for p in fn.params:
            pt = self.resolve_type(p.type_ann)
            param_types.append(pt)
            self.env.define(Symbol(p.name, pt, p.is_mut, p.span))
        ret = self.resolve_type(fn.return_type)
        old_ret = self._current_fn_return
        self._current_fn_return = ret
        self._check_block(fn.body)
        self._current_fn_return = old_ret
        self.env.pop()

    def _check_let(self, stmt: LetStmt):
        rhs_ty = self.infer(stmt.value)
        if stmt.type_ann:
            ann = self.resolve_type(stmt.type_ann)
            unified = ann.unify(rhs_ty)
            if unified is None:
                self.diag.error(
                    "E0030",
                    f"Type mismatch: expected `{ann}`, found `{rhs_ty}`",
                    stmt.span,
                    hints=[f"the annotation says {ann} but the value is {rhs_ty}"]
                )
            ty = ann
        else:
            ty = rhs_ty
        self.env.define(Symbol(stmt.name, ty, stmt.is_mut, stmt.span))

    def _check_assign(self, stmt: AssignStmt):
        if isinstance(stmt.target, Identifier):
            sym = self.env.lookup(stmt.target.name)
            if sym is None:
                self.diag.error("E0020",
                    f"Undefined variable `{stmt.target.name}`",
                    stmt.span)
            elif not sym.is_mut:
                self.diag.error("E0044",
                    f"Cannot assign to immutable binding `{stmt.target.name}`",
                    stmt.span,
                    hints=["declare with `let mut` to allow assignment"])
        rhs_ty = self.infer(stmt.value)

    def _check_return(self, stmt: ReturnStmt):
        if stmt.value:
            ty = self.infer(stmt.value)
            if self._current_fn_return:
                unified = self._current_fn_return.unify(ty)
                if unified is None:
                    self.diag.error(
                        "E0033",
                        f"Return type mismatch: expected `{self._current_fn_return}`, found `{ty}`",
                        stmt.span
                    )
        else:
            if self._current_fn_return and self._current_fn_return != UNIT:
                self.diag.error(
                    "E0033",
                    f"Missing return value; expected `{self._current_fn_return}`",
                    stmt.span
                )

    def _check_if(self, stmt: IfStmt):
        cond_ty = self.infer(stmt.condition)
        if cond_ty != BOOL:
            self.diag.warning(
                "E0030",
                f"Condition type is `{cond_ty}`, expected `bool`",
                stmt.span
            )
        self.env.push(); self._check_block(stmt.then_branch); self.env.pop()
        for cond, body in stmt.elif_branches:
            self.infer(cond)
            self.env.push(); self._check_block(body); self.env.pop()
        if stmt.else_branch:
            self.env.push(); self._check_block(stmt.else_branch); self.env.pop()

    # ── Expression inference (synthesise mode) ────────────────────────────────

    def infer(self, expr: Expression) -> Type:
        ty = self._infer_inner(expr)
        expr.resolved_type = ty
        return ty

    def _infer_inner(self, expr: Expression) -> Type:
        if isinstance(expr, Literal):
            return self._literal_type(expr)

        if isinstance(expr, Identifier):
            sym = self.env.lookup(expr.name)
            if sym is None:
                self.diag.error("E0020",
                    f"Undefined variable `{expr.name}`",
                    expr.span,
                    hints=[f"did you declare `{expr.name}` with `let`?"])
                return UnknownType()
            return sym.ty

        if isinstance(expr, BinaryOp):
            return self._infer_binop(expr)

        if isinstance(expr, UnaryOp):
            ty = self.infer(expr.operand)
            if expr.op == "not" and ty != BOOL:
                self.diag.error("E0030",
                    f"Operator `not` requires `bool`, found `{ty}`", expr.span)
            return ty

        if isinstance(expr, Call):
            return self._infer_call(expr)

        if isinstance(expr, BorrowExpr):
            inner = self.infer(expr.expr)
            return RefType(inner, expr.mutable)

        if isinstance(expr, DerefExpr):
            inner = self.infer(expr.expr)
            if isinstance(inner, RefType):
                return inner.inner
            self.diag.error("E0030",
                f"Cannot dereference non-reference type `{inner}`", expr.span)
            return UnknownType()

        if isinstance(expr, AwaitExpr):
            inner = self.infer(expr.expr)
            if isinstance(inner, FutureType):
                return inner.inner
            self.diag.error("E0030",
                f"Cannot await non-future type `{inner}`", expr.span)
            return UnknownType()

        if isinstance(expr, ListLiteral):
            if not expr.elements:
                elem = self.fresh_var("elem")
            else:
                elem = self.infer(expr.elements[0])
                for e in expr.elements[1:]:
                    et = self.infer(e)
                    if et != elem:
                        self.diag.error("E0030",
                            f"List element type mismatch: `{elem}` vs `{et}`",
                            e.span)
            return GenericType("Vec", [elem])

        if isinstance(expr, ModelCall):
            return FutureType(UnknownType())

        return UnknownType()

    def _literal_type(self, lit: Literal) -> Type:
        v = lit.value
        if isinstance(v, bool):   return BOOL   # bool before int (bool is int subtype)
        if isinstance(v, int):    return I32
        if isinstance(v, float):  return F32
        if isinstance(v, str):    return STR
        return UnknownType()

    def _infer_binop(self, expr: BinaryOp) -> Type:
        lt = self.infer(expr.left)
        rt = self.infer(expr.right)
        op = expr.op

        # Comparison operators → bool
        if op in ("==", "!=", "<", ">", "<=", ">="):
            unified = lt.unify(rt)
            if unified is None:
                self.diag.error("E0030",
                    f"Cannot compare `{lt}` with `{rt}` using `{op}`",
                    expr.span)
            return BOOL

        # Boolean operators
        if op in ("and", "or"):
            if lt != BOOL:
                self.diag.error("E0030", f"Expected `bool`, found `{lt}`", expr.left.span)
            if rt != BOOL:
                self.diag.error("E0030", f"Expected `bool`, found `{rt}`", expr.right.span)
            return BOOL

        # Arithmetic operators
        if op in ("+", "-", "*", "/", "%"):
            # String concatenation via +
            if op == "+" and lt == STR and rt == STR:
                return STR
            promoted = numeric_promotion(lt, rt)
            if promoted:
                return promoted
            unified = lt.unify(rt)
            if unified:
                return unified
            self.diag.error("E0030",
                f"Cannot apply `{op}` to `{lt}` and `{rt}`", expr.span)
            return UnknownType()

        return UnknownType()

    def _infer_call(self, expr: Call) -> Type:
        fn_ty = self.infer(expr.callee)
        if isinstance(fn_ty, FunctionType):
            if len(expr.args) != len(fn_ty.params):
                self.diag.error("E0032",
                    f"Expected {len(fn_ty.params)} argument(s), found {len(expr.args)}",
                    expr.span)
            for arg, expected in zip(expr.args, fn_ty.params):
                arg_ty = self.infer(arg)
                if arg_ty.unify(expected) is None:
                    self.diag.error("E0030",
                        f"Argument type mismatch: expected `{expected}`, found `{arg_ty}`",
                        arg.span)
            ret = fn_ty.return_type
            return FutureType(ret) if fn_ty.is_async else ret
        # Unknown callee (e.g. builtin)
        for arg in expr.args:
            self.infer(arg)
        return UnknownType()

    def _iter_element_type(self, ty: Type) -> Type:
        if isinstance(ty, GenericType) and ty.name in ("Vec", "List", "Array"):
            return ty.args[0] if ty.args else self.fresh_var("elem")
        return self.fresh_var("elem")
