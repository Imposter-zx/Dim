# dim_mir_to_llvm.py — MIR to LLVM IR Codegen
from typing import Dict, List, Optional, Any, Tuple
from dim_mir import (
    MIRModule,
    MIRFunction,
    BasicBlock,
    MIRStatement,
    Terminator,
    Assign,
    StorageLive,
    StorageDead,
    Borrow,
    Drop,
    Goto,
    Branch,
    Return,
    Call as MIRCall,
    PromptCall,
    ConstOperand,
    PlaceOperand,
    BorrowOperand,
    UseRValue,
    BinOpRValue,
    UnOpRValue,
    TensorRValue,
    Operand,
    Local,
    Place,
    Mutability,
    BorrowKind,
)
from dim_types import (
    Type,
    I32,
    I64,
    F32,
    F64,
    BOOL,
    STR,
    UNIT,
    TensorType,
    PromptType,
    GenericType,
)


class LLVMGenerator:
    def __init__(self):
        self._output: List[str] = []
        self._local_names: Dict[int, str] = {}
        self._block_names: Dict[int, str] = {}
        self._const_counter = 0
        self._current_fn: Optional[MIRFunction] = None

    def generate(self, module: MIRModule) -> str:
        self._output = [
            "; Dim-compiled module: " + module.name,
            'target triple = "x86_64-pc-linux-gnu"',
            'target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"',
            "",
            "; Runtime declarations",
            "declare i8* @dim_runtime_prompt_call(i8*, i8*, i1)",
            "declare void @dim_runtime_print_i32(i32)",
            "declare void @dim_runtime_panic(i8*)",
            "",
        ]
        for fn in module.functions:
            self._gen_function_decl(fn)
        self._output.append("")
        for fn in module.functions:
            self._gen_function(fn)
        return "\n".join(self._output)

    def _gen_function_decl(self, fn: MIRFunction):
        ret_ll = self._type_to_llvm(fn.return_type)
        params: List[str] = []
        for p in fn.params:
            ll_ty = self._type_to_llvm(p.ty)
            params.append(ll_ty)
        self._output.append(f"declare {ret_ll} @{fn.name}({', '.join(params)})")

    def _gen_function(self, fn: MIRFunction):
        self._current_fn = fn
        self._local_names.clear()
        self._block_names.clear()
        self._const_counter = 0

        for idx, loc in enumerate(fn.locals):
            name = loc.name or ("_local_" + str(loc.index))
            self._local_names[loc.index] = "%" + name

        ret_ll = self._type_to_llvm(fn.return_type)
        params: List[str] = []
        for p in fn.params:
            p_name = self._local_names.get(
                p.index, "%" + (p.name or ("p" + str(p.index)))
            )
            ll_ty = self._type_to_llvm(p.ty)
            params.append(ll_ty + " " + p_name)

        self._output.append(f"define {ret_ll} @{fn.name}({', '.join(params)}) {{")

        for idx, bb in enumerate(fn.blocks):
            block_name = "bb" + str(idx)
            self._block_names[id(bb)] = block_name
            self._output.append(f"{block_name}:")

            for stmt in bb.stmts:
                self._gen_stmt(stmt)

            if bb.terminator:
                self._gen_terminator(bb.terminator, fn.return_type)
            elif fn.return_type in (UNIT, None):
                self._output.append("  ret void")
            else:
                self._output.append(f"  br label %bb0")

        self._output.append("}")
        self._output.append("")

    def _gen_stmt(self, stmt: MIRStatement):
        if isinstance(stmt, StorageLive):
            pass

        elif isinstance(stmt, StorageDead):
            pass

        elif isinstance(stmt, Assign):
            dest_name = self._place_to_llvm(stmt.place)
            val, val_llty = self._rvalue_to_llvm(stmt.rvalue)
            if val and dest_name:
                self._output.append(f"  {dest_name} = {val}")

        elif isinstance(stmt, Borrow):
            dest_name = self._place_to_llvm(stmt.dest)
            src_name = self._place_to_llvm(stmt.place)
            llty = self._type_to_llvm(stmt.place.local.ty)
            ptr_str = f"  {dest_name} = alloca {llty}"
            self._output.append(ptr_str)
            self._output.append(f"  store {llty} {src_name}, ptr {dest_name}")

        elif isinstance(stmt, Drop):
            ptr_name = self._place_to_llvm(stmt.place)
            llty = self._type_to_llvm(stmt.place.local.ty)
            self._output.append(f"  ; drop {ptr_name} ({llty})")

    def _gen_terminator(self, term: Terminator, ret_ty: Any):
        if isinstance(term, Return):
            if term.value and ret_ty != UNIT:
                val = self._op_to_llvm(term.value)
                llty = self._type_to_llvm(getattr(term.value, "ty", None) or ret_ty)
                self._output.append(f"  ret {llty} {val}")
            else:
                self._output.append("  ret void")

        elif isinstance(term, Goto):
            target = self._block_names.get(term.target, "bb" + str(term.target))
            self._output.append(f"  br label %{target}")

        elif isinstance(term, Branch):
            cond = self._op_to_llvm(term.condition)
            true_t = self._block_names.get(
                term.true_target, "bb" + str(term.true_target)
            )
            false_t = self._block_names.get(
                term.false_target, "bb" + str(term.false_target)
            )
            self._output.append(f"  br i1 {cond}, label %{true_t}, label %{false_t}")

        elif isinstance(term, MIRCall):
            args_strs = [self._op_to_llvm(a) for a in term.args]
            args_lltys = [self._type_to_llvm(getattr(a, "ty", None)) for a in term.args]
            call_sig = ", ".join(f"{t} {v}" for t, v in zip(args_lltys, args_strs))
            callee_str = (
                term.callee
                if isinstance(term.callee, str)
                else self._op_to_llvm(term.callee)
            )
            ret_llty = self._type_to_llvm(term.dest.local.ty) if term.dest else "void"
            if term.dest:
                dest_name = self._place_to_llvm(term.dest)
                self._output.append(
                    f"  {dest_name} = call {ret_llty} @{callee_str}({call_sig})"
                )
            else:
                self._output.append(f"  call {ret_llty} @{callee_str}({call_sig})")
            next_t = self._block_names.get(
                id(term.next_block), "bb" + str(term.next_block.id)
            )
            self._output.append(f"  br label %{next_t}")

        elif isinstance(term, PromptCall):
            input_val = self._op_to_llvm(term.input)
            llty = self._type_to_llvm(getattr(term.input, "ty", None))
            self._output.append(f"  ; AI call: {term.prompt_ref}")
            if term.dest:
                dest_name = self._place_to_llvm(term.dest)
                call_str = f"  {dest_name} = call i8* @dim_runtime_prompt_call(i8* null, {llty} {input_val}, i1 {'1' if term.deterministic else '0'})"
                self._output.append(call_str)
            else:
                self._output.append(
                    f"  call i8* @dim_runtime_prompt_call(i8* null, {llty} {input_val}, i1 {'1' if term.deterministic else '0'})"
                )
            next_t = self._block_names.get(
                id(term.next_block), "bb" + str(term.next_block.id)
            )
            self._output.append(f"  br label %{next_t}")

    def _type_to_llvm(self, ty: Any) -> str:
        if ty is None:
            return "void"
        if ty == I32:
            return "i32"
        if ty == I64:
            return "i64"
        if ty == F32:
            return "float"
        if ty == F64:
            return "double"
        if ty == BOOL:
            return "i1"
        if ty == STR:
            return "i8*"
        if ty == UNIT:
            return "void"
        if isinstance(ty, TensorType):
            return f"<{len(ty.shape)} x float>"
        if isinstance(ty, PromptType):
            return "i8*"
        if isinstance(ty, GenericType):
            return "ptr"
        from dim_types import TypeVar

        if isinstance(ty, TypeVar):
            return "void"
        return "i8*"

    def _op_to_llvm(self, op: Any) -> str:
        if isinstance(op, ConstOperand):
            if op.value is True:
                return "1"
            if op.value is False:
                return "0"
            if isinstance(op.value, str):
                return f'c"{op.value}\00"'
            if isinstance(op.value, float):
                import struct

                b = struct.pack("f", op.value)
                hex_b = b.hex()
                return f"float({op.value})"
            if isinstance(op.value, int):
                return str(op.value)
            return "0"
        if isinstance(op, PlaceOperand):
            return self._place_to_llvm(op.place)
        if isinstance(op, BorrowOperand):
            ptr = self._place_to_llvm(op.place)
            return f"load {self._type_to_llvm(op.place.local.ty)}, ptr {ptr}"
        return "0"

    def _place_to_llvm(self, place: Place) -> str:
        if place.local.index in self._local_names:
            return self._local_names[place.local.index]
        name = place.local.name or ("_local_" + str(place.local.index))
        return "%" + name

    def _rvalue_to_llvm(self, rval: Any) -> Tuple[str, str]:
        from dim_mir import UseRValue, BinOpRValue, UnOpRValue, TensorRValue

        if isinstance(rval, UseRValue):
            return self._op_to_llvm(rval.operand), self._type_to_llvm(
                getattr(rval.operand, "ty", None)
            )

        if isinstance(rval, BinOpRValue):
            llty = self._type_to_llvm(getattr(rval.left, "ty", None))
            lval = self._op_to_llvm(rval.left)
            rval_str = self._op_to_llvm(rval.right)
            op_map = {
                "+": f"add {llty} {lval}, {rval_str}",
                "-": f"sub {llty} {lval}, {rval_str}",
                "*": f"mul {llty} {lval}, {rval_str}",
                "/": f"sdiv {llty} {lval}, {rval_str}",
                "%": f"srem {llty} {lval}, {rval_str}",
                "==": f"icmp eq {llty} {lval}, {rval_str}",
                "!=": f"icmp ne {llty} {lval}, {rval_str}",
                "<": f"icmp slt {llty} {lval}, {rval_str}",
                ">": f"icmp sgt {llty} {lval}, {rval_str}",
                "<=": f"icmp sle {llty} {lval}, {rval_str}",
                ">=": f"icmp sge {llty} {lval}, {rval_str}",
            }
            instr = op_map.get(rval.op, f"add {llty} {lval}, {rval_str}")
            return instr, llty

        if isinstance(rval, UnOpRValue):
            llty = self._type_to_llvm(getattr(rval.operand, "ty", None))
            val = self._op_to_llvm(rval.operand)
            if rval.op == "-":
                return f"sub {llty} zeroinitializer, {val}", llty
            if rval.op in ("not", "!"):
                return f"xor {llty} {val}, 1", llty
            return f"{rval.op} {llty} {val}", llty

        if isinstance(rval, TensorRValue):
            elem_count = 1
            for d in rval.shape:
                elem_count *= d
            llty = f"<{elem_count} x float>"
            return f"zeroinitializer", llty

        return "undef", "i32"
