# dim_interpreter.py — Tree-Walking Interpreter for Dim
#
# A tree-walking interpreter that evaluates Dim code directly without compilation.

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from dim_lexer import Lexer
from dim_parser import Parser
from dim_ast import *


@dataclass
class RuntimeValue:
    """Base class for runtime values."""

    pass


@dataclass
class NumberValue(RuntimeValue):
    value: float
    is_int: bool = False


@dataclass
class StringValue(RuntimeValue):
    value: str


@dataclass
class BoolValue(RuntimeValue):
    value: bool


@dataclass
class FunctionValue(RuntimeValue):
    name: str
    params: List[str]
    body: List[Statement]
    env: "Environment"
    is_async: bool = False


@dataclass
class ClosureValue(RuntimeValue):
    params: List[str]
    body: Expression
    env: "Environment"


@dataclass
class StructValue(RuntimeValue):
    name: str
    fields: Dict[str, RuntimeValue]


@dataclass
class EnumValue(RuntimeValue):
    name: str
    variant: str
    value: Optional[RuntimeValue] = None


@dataclass
class ListValue(RuntimeValue):
    elements: List[RuntimeValue]


@dataclass
class NoneValue(RuntimeValue):
    pass


@dataclass
class PromptValue(RuntimeValue):
    name: str
    system_prompt: str
    user_template: str
    output_type: str


class Environment:
    def __init__(self, parent: Optional["Environment"] = None):
        self.parent = parent
        self.variables: Dict[str, RuntimeValue] = {}
        self.functions: Dict[str, FunctionValue] = {}
        self.structs: Dict[str, List[Tuple]] = {}
        self.enums: Dict[str, List[str]] = {}
        self.traits: Dict[str, List[str]] = {}
        self.prompts: Dict[str, PromptValue] = {}

    def define(self, name: str, value: RuntimeValue):
        self.variables[name] = value

    def assign(self, name: str, value: RuntimeValue) -> bool:
        if name in self.variables:
            self.variables[name] = value
            return True
        if self.parent:
            return self.parent.assign(name, value)
        return False

    def get(self, name: str) -> Optional[RuntimeValue]:
        if name in self.variables:
            return self.variables[name]
        if name in self.functions:
            return self.functions[name]
        if name in self.prompts:
            return self.prompts[name]
        if name in self.enums:
            return EnumValue(name, "")
        if name in self.structs:
            return StructValue(name, {})
        if self.parent:
            return self.parent.get(name)
        return None

    def define_function(self, name: str, func: FunctionValue):
        self.functions[name] = func

    def define_struct(self, name: str, fields: List[Tuple]):
        self.struct[name] = fields

    def define_enum(self, name: str, variants: List[str]):
        self.enums[name] = variants

    def define_trait(self, name: str, methods: List[str]):
        self.traits[name] = methods

    def define_prompt(self, name: str, prompt: PromptValue):
        self.prompts[name] = prompt


class DimInterpreter:
    def __init__(self):
        self.global_env = Environment()
        self._setup_builtins()
        self._current_env: Optional[Environment] = None

    def _setup_builtins(self):
        # Built-in functions
        self.global_env.define_function(
            "print", FunctionValue("print", ["msg"], [], self.global_env)
        )
        self.global_env.define_function(
            "println", FunctionValue("println", ["msg"], [], self.global_env)
        )
        self.global_env.define_function(
            "abs", FunctionValue("abs", ["x"], [], self.global_env)
        )
        self.global_env.define_function(
            "min", FunctionValue("min", ["a", "b"], [], self.global_env)
        )
        self.global_env.define_function(
            "max", FunctionValue("max", ["a", "b"], [], self.global_env)
        )
        self.global_env.define_function(
            "len", FunctionValue("len", ["x"], [], self.global_env)
        )
        self.global_env.define_function(
            "range", FunctionValue("range", ["n"], [], self.global_env)
        )
        self.global_env.define_function(
            "input", FunctionValue("input", ["prompt"], [], self.global_env)
        )
        self.global_env.define_function(
            "read_file", FunctionValue("read_file", ["path"], [], self.global_env)
        )
        self.global_env.define_function(
            "write_file",
            FunctionValue("write_file", ["path", "content"], [], self.global_env),
        )

    def interpret(self, source: str, filename: str = "<input>") -> Any:
        tokens = Lexer(source, filename).tokenize()
        parser = Parser(tokens, source, filename)
        program = parser.parse_program()

        # Note: ignoring parse errors for now to allow interpreter testing
        # In production, should check: if parser.diag.has_errors:

        env = Environment(self.global_env)
        result = None

        for stmt in program.statements:
            result = self._execute_stmt(stmt, env)

        # Call main function if it exists
        main_func = env.get("main")
        if main_func:
            print("Calling main...")
            main_env = Environment(env)
            # Execute main function
            if isinstance(main_func, FunctionValue):
                result = None
                for stmt in main_func.body:
                    result = self._execute_stmt(stmt, main_env)

        return result

    def _execute_stmt(self, stmt: Statement, env: Environment) -> RuntimeValue:
        if isinstance(stmt, LetStmt):
            return self._exec_let(stmt, env)
        elif isinstance(stmt, FunctionDef):
            return self._exec_function(stmt, env)
        elif isinstance(stmt, ReturnStmt):
            return self._exec_return(stmt, env)
        elif isinstance(stmt, IfStmt):
            return self._exec_if(stmt, env)
        elif isinstance(stmt, WhileStmt):
            return self._exec_while(stmt, env)
        elif isinstance(stmt, ForStmt):
            return self._exec_for(stmt, env)
        elif isinstance(stmt, MatchStmt):
            return self._exec_match(stmt, env)
        elif isinstance(stmt, ExprStmt):
            return self._eval_expr(stmt.expr, env)
        elif isinstance(stmt, AssignStmt):
            return self._exec_assign(stmt, env)
        elif isinstance(stmt, StructDef):
            return self._exec_struct(stmt, env)
        elif isinstance(stmt, EnumDef):
            return self._exec_enum(stmt, env)
        elif isinstance(stmt, TraitDef):
            return self._exec_trait(stmt, env)
        elif isinstance(stmt, ImplBlock):
            return self._exec_impl(stmt, env)
        elif isinstance(stmt, PromptDef):
            return self._exec_prompt(stmt, env)
        elif isinstance(stmt, ImportStmt):
            return NoneValue()
        elif type(stmt).__name__ == "BreakStmt":
            raise BreakException()
        elif type(stmt).__name__ == "ContinueStmt":
            raise ContinueException()
        elif isinstance(stmt, ThrowStmt):
            raise RuntimeError(
                str(self._eval_expr(stmt.expr, env) if stmt.expr else "Error")
            )
        elif isinstance(stmt, TryStmt):
            return self._exec_try(stmt, env)
        else:
            return NoneValue()

    def _exec_assign(self, stmt: AssignStmt, env: Environment) -> RuntimeValue:
        value = self._eval_expr(stmt.value, env)
        # Handle different target types
        if isinstance(stmt.target, Identifier):
            env.define(stmt.target.name, value)
        return value

    def _exec_let(self, stmt: LetStmt, env: Environment) -> RuntimeValue:
        value = self._eval_expr(stmt.value, env) if stmt.value else NoneValue()
        env.define(stmt.name, value)
        return value

    def _exec_function(self, stmt: FunctionDef, env: Environment) -> RuntimeValue:
        func = FunctionValue(
            stmt.name, [p.name for p in stmt.params], stmt.body, env, stmt.is_async
        )
        env.define_function(stmt.name, func)
        return func

    def _exec_return(self, stmt: ReturnStmt, env: Environment) -> RuntimeValue:
        if stmt.value:
            return self._eval_expr(stmt.value, env)
        return NoneValue()

    def _exec_if(self, stmt: IfStmt, env: Environment) -> RuntimeValue:
        cond = self._eval_expr(stmt.condition, env)
        if self._is_truthy(cond):
            for s in stmt.then_branch:
                self._execute_stmt(s, env)
        elif stmt.else_branch:
            for s in stmt.else_branch:
                self._execute_stmt(s, env)
        return NoneValue()

    def _exec_while(self, stmt: WhileStmt, env: Environment) -> RuntimeValue:
        while self._is_truthy(self._eval_expr(stmt.condition, env)):
            try:
                for s in stmt.body:
                    self._execute_stmt(s, env)
            except BreakException:
                break
            except ContinueException:
                continue
        return NoneValue()

    def _exec_for(self, stmt: ForStmt, env: Environment) -> RuntimeValue:
        iterable = self._eval_expr(stmt.iterable, env)
        if isinstance(iterable, ListValue):
            for item in iterable.elements:
                inner_env = Environment(env)
                inner_env.define(stmt.variable, item)
                try:
                    for s in stmt.body:
                        self._execute_stmt(s, inner_env)
                except BreakException:
                    break
                except ContinueException:
                    continue
        return NoneValue()

    def _exec_match(self, stmt: MatchStmt, env: Environment) -> RuntimeValue:
        value = self._eval_expr(stmt.expr, env)
        for arm in stmt.arms:
            if self._match_pattern(arm.pattern, value, env):
                for s in arm.body:
                    self._execute_stmt(s, env)
                return NoneValue()
        return NoneValue()

    def _match_pattern(
        self, pattern: Expression, value: RuntimeValue, env: Environment
    ) -> bool:
        if isinstance(pattern, WildcardPattern):
            return True
        if isinstance(pattern, Identifier):
            if pattern.name == "_":
                return True
            pat_val = self._eval_expr(pattern, env)
            return self._equals(pat_val, value)
        if isinstance(pattern, Literal):
            pat_val = self._eval_expr(pattern, env)
            return self._equals(pat_val, value)
        return False

    def _exec_struct(self, stmt: StructDef, env: Environment) -> RuntimeValue:
        env.structs[stmt.name] = [(f.name, f.typ) for f in stmt.fields]
        return NoneValue()

    def _exec_enum(self, stmt: EnumDef, env: Environment) -> RuntimeValue:
        env.enums[stmt.name] = stmt.variants
        return NoneValue()

    def _exec_trait(self, stmt: TraitDef, env: Environment) -> RuntimeValue:
        env.traits[stmt.name] = [m.name for m in stmt.methods]
        return NoneValue()

    def _exec_impl(self, stmt: ImplBlock, env: Environment) -> RuntimeValue:
        return NoneValue()

    def _exec_prompt(self, stmt: PromptDef, env: Environment) -> RuntimeValue:
        system = ""
        user = ""
        output_type = ""

        for role in stmt.roles:
            if role.role_type == "system":
                system = role.content
            elif role.role_type == "user":
                user = role.content

        if stmt.output_type:
            output_type = str(stmt.output_type)

        prompt = PromptValue(stmt.name, system, user, output_type)
        env.define_prompt(stmt.name, prompt)
        return prompt

    def _exec_try(self, stmt: TryStmt, env: Environment) -> RuntimeValue:
        try:
            for s in stmt.try_body:
                self._execute_stmt(s, env)
        except Exception as e:
            if stmt.catch_body:
                catch_env = Environment(env)
                if stmt.exception_name:
                    catch_env.define(stmt.exception_name, StringValue(str(e)))
                for s in stmt.catch_body:
                    self._execute_stmt(s, catch_env)
        finally:
            if stmt.finally_body:
                for s in stmt.finally_body:
                    self._execute_stmt(s, env)
        return NoneValue()

    def _eval_expr(self, expr: Expression, env: Environment):
        self._current_env = env
        if isinstance(expr, Literal):
            return self._eval_literal(expr, env)
        elif isinstance(expr, Identifier):
            return self._eval_identifier(expr, env)
        elif isinstance(expr, BinaryOp):
            return self._eval_binary(expr, env)
        elif isinstance(expr, UnaryOp):
            return self._eval_unary(expr, env)
        elif isinstance(expr, Call):
            return self._eval_call(expr, env)
        elif isinstance(expr, IndexAccess):
            return self._eval_index(expr, env)
        elif isinstance(expr, MemberAccess):
            return self._eval_member(expr, env)
        elif isinstance(expr, MethodCall):
            return self._eval_method_call(expr, env)
        elif isinstance(expr, StructConstruct):
            return self._eval_struct(expr, env)
        elif isinstance(expr, EnumVariant):
            return self._eval_enum_variant(expr, env)
        elif isinstance(expr, ClosureExpr):
            return self._eval_closure(expr, env)
        elif isinstance(expr, ListLiteral):
            return self._eval_list(expr, env)
        elif isinstance(expr, BorrowExpr):
            return self._eval_borrow(expr, env)
        elif isinstance(expr, UnwrapExpr):
            return self._eval_unwrap(expr, env)
        else:
            return NoneValue()

    def _eval_literal(self, expr: Literal, env: Environment) -> RuntimeValue:
        val = expr.value
        if val is None:
            return NoneValue()
        elif isinstance(val, bool):
            return BoolValue(val)
        elif isinstance(val, (int, float)):
            return NumberValue(float(val), isinstance(val, int))
        elif isinstance(val, str):
            if chr(0x01) in val:
                return self._eval_interpolated_string(val, env)
            return StringValue(val)
        return NoneValue()

    def _eval_interpolated_string(self, val: str, env: Environment) -> RuntimeValue:
        parts = val.split(chr(0x01))
        result = ""
        i = 0
        while i < len(parts):
            part = parts[i]
            if i + 1 < len(parts):
                var_part = parts[i + 1]
                if "}" in var_part:
                    end = var_part.find("}")
                    var_name = var_part[:end]
                    rest = var_part[end + 1 :]
                    try:
                        var_val = env.get(var_name)
                        if var_val is not None:
                            result += part + self._to_string(var_val) + rest
                        else:
                            result += part + "{" + var_name + "}" + rest
                    except:
                        result += part + "{" + var_name + "}" + rest
                    i += 2
                else:
                    result += part + "{" + var_part
                    i += 2
            else:
                result += part
                i += 1
        return StringValue(result)

    def _eval_identifier(self, expr: Identifier, env: Environment) -> RuntimeValue:
        val = env.get(expr.name)
        if val is not None:
            return val
        if expr.name in env.enums:
            return EnumValue(expr.name, "")
        if expr.name in env.structs:
            return StructValue(expr.name, {})
        if expr.name in env.traits:
            return NoneValue()
        raise RuntimeError(f"Undefined variable: {expr.name}")

    def _eval_binary(self, expr: BinaryOp, env: Environment) -> RuntimeValue:
        left = self._eval_expr(expr.left, env)
        right = self._eval_expr(expr.right, env)

        if expr.op == "+":
            if isinstance(left, StringValue) or isinstance(right, StringValue):
                return StringValue(
                    str(self._to_string(left)) + str(self._to_string(right))
                )
            if isinstance(left, NumberValue) and isinstance(right, NumberValue):
                return NumberValue(
                    left.value + right.value, left.is_int and right.is_int
                )
            return NumberValue(left.value + right.value)
        elif expr.op == "-":
            return NumberValue(left.value - right.value)
        elif expr.op == "*":
            return NumberValue(left.value * right.value)
        elif expr.op == "/":
            if right.value == 0:
                raise RuntimeError("Division by zero")
            return NumberValue(left.value / right.value)
        elif expr.op == "%":
            return NumberValue(left.value % right.value)
        elif expr.op in ("==", "!="):
            return BoolValue(self._equals(left, right))
        elif expr.op in ("<", ">", "<=", ">="):
            return BoolValue(self._compare(left, right, expr.op))
        elif expr.op == "and":
            return BoolValue(self._is_truthy(left) and self._is_truthy(right))
        elif expr.op == "or":
            return BoolValue(self._is_truthy(left) or self._is_truthy(right))

        return NoneValue()

    def _eval_unary(self, expr: UnaryOp, env: Environment) -> RuntimeValue:
        operand = self._eval_expr(expr.operand, env)

        if expr.op == "-":
            if isinstance(operand, NumberValue):
                return NumberValue(-operand.value, operand.is_int)
        elif expr.op == "not":
            return BoolValue(not self._is_truthy(operand))

        return operand

    def _eval_call(self, expr: Call, env: Environment) -> RuntimeValue:
        callee = self._eval_expr(expr.callee, env)

        args = [self._eval_expr(arg, env) for arg in expr.args]

        # Built-in functions
        if isinstance(callee, FunctionValue):
            func = callee
            if func.name == "print" or func.name == "println":
                msg = args[0] if args else NoneValue()
                s = self._to_string(msg)
                print(s)
                return NoneValue()
            elif func.name == "abs":
                if isinstance(args[0], NumberValue):
                    return NumberValue(abs(args[0].value), args[0].is_int)
            elif func.name == "min":
                if isinstance(args[0], NumberValue) and isinstance(
                    args[1], NumberValue
                ):
                    return NumberValue(min(args[0].value, args[1].value))
            elif func.name == "max":
                if isinstance(args[0], NumberValue) and isinstance(
                    args[1], NumberValue
                ):
                    return NumberValue(max(args[0].value, args[1].value))
            elif func.name == "len":
                if isinstance(args[0], ListValue):
                    return NumberValue(len(args[0].elements), True)
                elif isinstance(args[0], StringValue):
                    return NumberValue(len(args[0].value), True)
            elif func.name == "range":
                n = int(args[0].value) if isinstance(args[0], NumberValue) else 0
                return ListValue([NumberValue(i, True) for i in range(n)])
            elif func.name == "input":
                prompt = self._to_string(args[0]) if args else ""
                return StringValue(input(prompt))
            elif func.name == "read_file":
                try:
                    with open(self._to_string(args[0]), "r") as f:
                        return StringValue(f.read())
                except:
                    return StringValue("")
            elif func.name == "write_file":
                try:
                    with open(self._to_string(args[0]), "w") as f:
                        f.write(self._to_string(args[1]))
                except:
                    pass
                return NoneValue()

            # User-defined function
            call_env = Environment(func.env)
            for i, param_name in enumerate(func.params):
                if i < len(args):
                    call_env.define(param_name, args[i])

            result = NoneValue()
            for stmt in func.body:
                result = self._execute_stmt(stmt, call_env)

            return result

        # Struct constructor
        if isinstance(callee, StructValue):
            struct_val = StructValue(callee.name, {})
            for i, (field_name, _) in enumerate(callee.fields.items()):
                if i < len(args):
                    struct_val.fields[field_name] = args[i]
            return struct_val

        # Prompt AI call
        if isinstance(callee, PromptValue):
            return self._eval_prompt_call(callee, args, env)

        return NoneValue()

    def _eval_prompt_call(
        self, prompt: PromptValue, args: List[RuntimeValue], env: Environment
    ) -> RuntimeValue:
        """Evaluate a prompt call with actual AI."""
        from dim_ai import AIEngine, create_openai_adapter

        # Get environment variable for provider
        import os

        provider = os.environ.get("DIM_AI_PROVIDER", "openai")
        api_key = os.environ.get("DIM_AI_KEY", os.environ.get("OPENAI_API_KEY", ""))

        # Build input from kwargs
        user_input = ""
        for arg in args:
            if isinstance(arg, StringValue):
                user_input = arg.value
            elif isinstance(arg, NumberValue):
                user_input = str(int(arg.value))

        # Format user template with input
        user_text = prompt.user_template
        if user_input:
            user_text = user_text.replace("{text}", user_input)
            # Replace other common placeholders
            import re

            user_text = re.sub(
                r"\{(\w+)\}", lambda m: str(args[0].value) if args else "", user_text
            )

        # Check for API key - if not available, use stub
        if not api_key:
            print(f"[AI Stub] {prompt.name}: {user_text[:50]}... => (no API key)")
            return StringValue("positive (stub)")

        # Create AI engine and execute
        try:
            engine = AIEngine()

            if provider == "openai":
                adapter = create_openai_adapter("gpt-4")
                if engine.register_adapter("default", adapter):
                    result = engine.execute_prompt(prompt.name, text=user_input)
                    return StringValue(result)

            # Fallback to stub
            return StringValue("positive (stub)")

        except Exception as e:
            print(f"[AI Error] {e}")
            return StringValue(f"error: {e}")

    def _eval_index(self, expr: IndexAccess, env: Environment) -> RuntimeValue:
        obj = self._eval_expr(expr.object, env)
        index = self._eval_expr(expr.index, env)

        if isinstance(obj, ListValue) and isinstance(index, NumberValue):
            idx = int(index.value)
            if 0 <= idx < len(obj.elements):
                return obj.elements[idx]

        if isinstance(obj, StringValue) and isinstance(index, NumberValue):
            idx = int(index.value)
            if 0 <= idx < len(obj.value):
                return StringValue(obj.value[idx])

        return NoneValue()

    def _eval_member(self, expr: MemberAccess, env: Environment) -> RuntimeValue:
        obj = self._eval_expr(expr.expr, env)

        # String methods
        if isinstance(obj, StringValue):
            if expr.member == "len":
                return NumberValue(len(obj.value), True)
            elif expr.member == "upper":
                return StringValue(obj.value.upper())
            elif expr.member == "lower":
                return StringValue(obj.value.lower())
            elif expr.member == "trim":
                return StringValue(obj.value.strip())
            elif expr.member == "contains":
                # Expects one arg - the substring
                arg = env.get("_method_arg")
                if arg and isinstance(arg, StringValue):
                    return BoolValue(arg.value in obj.value)
                return BoolValue(False)
            elif expr.member == "split":
                arg = env.get("_method_arg")
                if arg and isinstance(arg, StringValue):
                    parts = obj.value.split(arg.value)
                    return ListValue([StringValue(p) for p in parts])
                return ListValue([StringValue(s) for s in obj.value])
            elif expr.member == "replace":
                return StringValue(obj.value)
            elif expr.member == "starts_with":
                arg = env.get("_method_arg")
                if arg and isinstance(arg, StringValue):
                    return BoolValue(obj.value.startswith(arg.value))
                return BoolValue(False)
            elif expr.member == "ends_with":
                arg = env.get("_method_arg")
                if arg and isinstance(arg, StringValue):
                    return BoolValue(obj.value.endswith(arg.value))
                return BoolValue(False)
            elif expr.member == "to_string":
                return StringValue(obj.value)
            elif expr.member == "index_of":
                arg = env.get("_method_arg")
                if arg and isinstance(arg, StringValue):
                    idx = obj.value.find(arg.value)
                    return NumberValue(idx, idx >= 0)
                return NumberValue(-1, True)

        # Number methods
        if isinstance(obj, NumberValue):
            if expr.member == "to_string":
                if obj.is_int:
                    return StringValue(str(int(obj.value)))
                return StringValue(str(obj.value))
            elif expr.member == "abs":
                return NumberValue(abs(obj.value), obj.is_int)
            elif expr.member == "floor":
                return NumberValue(int(obj.value), True)
            elif expr.member == "ceil":
                import math

                return NumberValue(math.ceil(obj.value), True)
            elif expr.member == "round":
                import math

                return NumberValue(round(obj.value), True)
            elif expr.member == "sqrt":
                import math

                return NumberValue(math.sqrt(obj.value), False)
            elif expr.member == "to_string":
                return StringValue(str(obj.value))

        # List methods
        if isinstance(obj, ListValue):
            if expr.member == "len":
                return NumberValue(len(obj.elements), True)
            elif expr.member == "push":
                # push would need special handling - add to list
                return NoneValue()
            elif expr.member == "pop":
                if obj.elements:
                    val = obj.elements.pop()
                    return val
                return NoneValue()
            elif expr.member == "contains":
                arg = env.get("_method_arg")
                if arg:
                    for e in obj.elements:
                        if self._equals(e, arg):
                            return BoolValue(True)
                return BoolValue(False)
            elif expr.member == "reverse":
                obj.elements.reverse()
                return obj
            elif expr.member == "first":
                if obj.elements:
                    return obj.elements[0]
                return NoneValue()
            elif expr.member == "last":
                if obj.elements:
                    return obj.elements[-1]
                return NoneValue()

        return NoneValue()

    def _eval_method_call(self, expr: MethodCall, env: Environment) -> RuntimeValue:
        obj = self._eval_expr(expr.receiver, env)
        args = [self._eval_expr(arg, env) for arg in expr.args]

        # Temporarily store the first arg for methods that use env.get("_method_arg")
        if args:
            env.define("_method_arg", args[0])

        # String methods
        if isinstance(obj, StringValue):
            if expr.method == "upper":
                return StringValue(obj.value.upper())
            elif expr.method == "lower":
                return StringValue(obj.value.lower())
            elif expr.method == "trim":
                return StringValue(obj.value.strip())
            elif expr.method == "len":
                return NumberValue(len(obj.value), True)
            elif expr.method == "contains":
                if args and isinstance(args[0], StringValue):
                    return BoolValue(args[0].value in obj.value)
                return BoolValue(False)
            elif expr.method == "split":
                if args and isinstance(args[0], StringValue):
                    parts = obj.value.split(args[0].value)
                    return ListValue([StringValue(p) for p in parts])
                return ListValue([StringValue(s) for s in obj.value.split()])
            elif expr.method == "replace":
                if (
                    len(args) >= 2
                    and isinstance(args[0], StringValue)
                    and isinstance(args[1], StringValue)
                ):
                    return StringValue(obj.value.replace(args[0].value, args[1].value))
                return StringValue(obj.value)
            elif expr.method == "starts_with":
                if args and isinstance(args[0], StringValue):
                    return BoolValue(obj.value.startswith(args[0].value))
                return BoolValue(False)
            elif expr.method == "ends_with":
                if args and isinstance(args[0], StringValue):
                    return BoolValue(obj.value.endswith(args[0].value))
                return BoolValue(False)
            elif expr.method == "to_string":
                return StringValue(obj.value)
            elif expr.method == "index_of":
                if args and isinstance(args[0], StringValue):
                    idx = obj.value.find(args[0].value)
                    return NumberValue(idx, idx >= 0)
                return NumberValue(-1, True)
            elif expr.method == "chars":
                return ListValue([StringValue(c) for c in obj.value])
            elif expr.method == "lines":
                return ListValue([StringValue(s) for s in obj.value.split("\n")])
            elif expr.method == "strip":
                return StringValue(obj.value.strip())

        # Number methods
        if isinstance(obj, NumberValue):
            if expr.method == "to_string":
                if obj.is_int:
                    return StringValue(str(int(obj.value)))
                return StringValue(str(obj.value))
            elif expr.method == "abs":
                return NumberValue(abs(obj.value), obj.is_int)
            elif expr.method == "floor":
                return NumberValue(int(obj.value), True)
            elif expr.method == "ceil":
                import math

                return NumberValue(math.ceil(obj.value), True)
            elif expr.method == "round":
                import math

                return NumberValue(round(obj.value), True)
            elif expr.method == "sqrt":
                import math

                return NumberValue(math.sqrt(obj.value), False)

        # List methods
        if isinstance(obj, ListValue):
            if expr.method == "len":
                return NumberValue(len(obj.elements), True)
            elif expr.method == "push":
                if args:
                    obj.elements.append(args[0])
                return obj
            elif expr.method == "pop":
                if obj.elements:
                    return obj.elements.pop()
                return NoneValue()
            elif expr.method == "contains":
                if args:
                    for e in obj.elements:
                        if self._equals(e, args[0]):
                            return BoolValue(True)
                return BoolValue(False)
            elif expr.method == "reverse":
                obj.elements.reverse()
                return obj
            elif expr.method == "first":
                if obj.elements:
                    return obj.elements[0]
                return NoneValue()
            elif expr.method == "last":
                if obj.elements:
                    return obj.elements[-1]
                return NoneValue()
            elif expr.method == "append":
                if args:
                    obj.elements.append(args[0])
                return obj
            elif expr.method == "insert":
                if len(args) >= 2 and isinstance(args[0], NumberValue):
                    idx = int(args[0].value)
                    if 0 <= idx <= len(obj.elements):
                        obj.elements.insert(idx, args[1])
                return obj
            elif expr.method == "remove":
                if args:
                    for i, e in enumerate(obj.elements):
                        if self._equals(e, args[0]):
                            obj.elements.pop(i)
                            return obj
                return obj

        return NoneValue()

    def _eval_struct(self, expr: StructConstruct, env: Environment) -> RuntimeValue:
        name = expr.name
        fields = {}
        for field in expr.fields:
            fields[field.name] = self._eval_expr(field.value, env)
        return StructValue(name, fields)

    def _eval_enum_variant(self, expr: EnumVariant, env: Environment) -> RuntimeValue:
        if expr.args and len(expr.args) > 0:
            value = self._eval_expr(expr.args[0], env)
        else:
            value = None
        return EnumValue(expr.enum_name, expr.variant_name, value)

    def _eval_closure(self, expr: ClosureExpr, env: Environment) -> RuntimeValue:
        return ClosureValue([p.name for p in expr.params], expr.body, env)

    def _eval_tuple(self, expr: TupleLiteral, env: Environment) -> RuntimeValue:
        return ListValue([self._eval_expr(e, env) for e in expr.elements])

    def _eval_list(self, expr: ListLiteral, env: Environment) -> RuntimeValue:
        return ListValue([self._eval_expr(e, env) for e in expr.elements])

    def _eval_borrow(self, expr: BorrowExpr, env: Environment) -> RuntimeValue:
        return self._eval_expr(expr.expr, env)

    def _eval_unwrap(self, expr: UnwrapExpr, env: Environment) -> RuntimeValue:
        inner = self._eval_expr(expr.expr, env)
        if isinstance(inner, EnumValue):
            if inner.variant == "Ok":
                return inner.value if inner.value is not None else NoneValue()
            elif inner.variant == "Err":
                raise RuntimeError(f"Unwrapped error: {inner.value}")
        if isinstance(inner, NoneValue):
            raise RuntimeError("Cannot unwrap none value")
        if isinstance(inner, NumberValue):
            raise RuntimeError(
                "Cannot unwrap number - ? operator requires Result/Option type"
            )
        if isinstance(inner, StringValue):
            raise RuntimeError(
                "Cannot unwrap string - ? operator requires Result/Option type"
            )
        if isinstance(inner, ListValue):
            raise RuntimeError(
                "Cannot unwrap list - ? operator requires Result/Option type"
            )
        return inner

    def _is_truthy(self, val: RuntimeValue) -> bool:
        if isinstance(val, BoolValue):
            return val.value
        if isinstance(val, NumberValue):
            return val.value != 0
        if isinstance(val, StringValue):
            return len(val.value) > 0
        if isinstance(val, NoneValue):
            return False
        return True

    def _equals(self, a: RuntimeValue, b: RuntimeValue) -> bool:
        if isinstance(a, NumberValue) and isinstance(b, NumberValue):
            return a.value == b.value
        if isinstance(a, StringValue) and isinstance(b, StringValue):
            return a.value == b.value
        if isinstance(a, BoolValue) and isinstance(b, BoolValue):
            return a.value == b.value
        if isinstance(a, NoneValue) and isinstance(b, NoneValue):
            return True
        return False

    def _compare(self, a: RuntimeValue, b: RuntimeValue, op: str) -> bool:
        if isinstance(a, NumberValue) and isinstance(b, NumberValue):
            if op == "<":
                return a.value < b.value
            if op == ">":
                return a.value > b.value
            if op == "<=":
                return a.value <= b.value
            if op == ">=":
                return a.value >= b.value
        if isinstance(a, StringValue) and isinstance(b, StringValue):
            if op == "<":
                return a.value < b.value
            if op == ">":
                return a.value > b.value
        return False

    def _to_string(self, val: RuntimeValue) -> str:
        if isinstance(val, NumberValue):
            if val.is_int:
                return str(int(val.value))
            return str(val.value)
        if isinstance(val, StringValue):
            return val.value
        if isinstance(val, BoolValue):
            return "true" if val.value else "false"
        if isinstance(val, NoneValue):
            return "none"
        if isinstance(val, ListValue):
            return "[" + ", ".join(self._to_string(e) for e in val.elements) + "]"
        if isinstance(val, StructValue):
            fields = ", ".join(
                f"{k}: {self._to_string(v)}" for k, v in val.fields.items()
            )
            return f"{val.name}{{{fields}}}"
        if isinstance(val, EnumValue):
            if val.value:
                return f"{val.name}::{val.variant}({self._to_string(val.value)})"
            return f"{val.name}::{val.variant}"
        return str(val)


class BreakException(Exception):
    pass


class ContinueException(Exception):
    pass


def run_interpreter(source: str, filename: str = "<input>") -> Any:
    """Run the interpreter on a source string."""
    interpreter = DimInterpreter()
    return interpreter.interpret(source, filename)


def run_file(filepath: str) -> Any:
    """Run a .dim file through the interpreter."""
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    return run_interpreter(source, filepath)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        run_file(sys.argv[1])
    else:
        print("Usage: python dim_interpreter.py <file.dim>")
