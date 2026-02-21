# dim_tests.py — Test Suite for Dim Compiler (Phase 1)
#
# Golden-file / unit tests that exercise the full pipeline:
# Lexer → Parser → Type Checker → MIR Lowering → Borrow Checker

from __future__ import annotations
import sys
import traceback
from dataclasses import dataclass
from typing import Callable, List, Optional


# ── Test harness ──────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    name:    str
    fn:      Callable[[], None]
    tags:    List[str]

_tests: List[TestCase] = []

def test(name: str = "", *tags: str):
    """Decorator to register a test function."""
    def decorator(fn: Callable):
        _tests.append(TestCase(name or fn.__name__, fn, list(tags)))
        return fn
    return decorator

def assert_eq(a, b, msg: str = ""):
    if a != b:
        raise AssertionError(f"{msg or 'assert_eq failed'}: {a!r} != {b!r}")

def assert_true(cond, msg: str = ""):
    if not cond:
        raise AssertionError(msg or "assert_true failed")

def assert_no_errors(diag_bag, msg: str = ""):
    if diag_bag.has_errors:
        errs = [str(d) for d in diag_bag.all if d.severity.name == "ERROR"]
        raise AssertionError(f"{msg or 'Expected no errors, but got'}: {errs}")

def assert_has_error(diag_bag, code: str):
    codes = [d.code for d in diag_bag.all]
    if code not in codes:
        raise AssertionError(f"Expected error {code}, found: {codes}")


def run_tests(filter_tag: Optional[str] = None):
    passed = failed = skipped = 0
    total  = len(_tests)

    print(f"\n{'='*60}")
    print(f"  Dim Compiler Test Suite — Phase 1")
    print(f"{'='*60}\n")

    for tc in _tests:
        if filter_tag and filter_tag not in tc.tags:
            skipped += 1
            continue
        try:
            tc.fn()
            print(f"  \033[32m✓\033[0m {tc.name}")
            passed += 1
        except AssertionError as e:
            print(f"  \033[31m✗\033[0m {tc.name}")
            print(f"      {e}")
            failed += 1
        except Exception as e:
            print(f"  \033[31m✗\033[0m {tc.name} [EXCEPTION]")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}\n")
    return failed == 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse(code: str):
    from dim_lexer import Lexer
    from dim_parser import Parser
    tokens = Lexer(code, "test.dim").tokenize()
    parser = Parser(tokens, code, "test.dim")
    return parser.parse_program(), parser.diag

def _type_check(code: str):
    from dim_lexer import Lexer
    from dim_parser import Parser
    from dim_semantic import SemanticAnalyzer
    tokens = Lexer(code, "test.dim").tokenize()
    parser = Parser(tokens, code, "test.dim")
    ast    = parser.parse_program()
    sem    = SemanticAnalyzer(code, "test.dim")
    ok     = sem.analyze(ast)
    return ast, sem.diag, ok

def _lower(code: str):
    from dim_mir_lowering import lower_program
    ast, diag, ok = _type_check(code)
    module = lower_program(ast)
    return module, diag


# ═══════════════════════════════════════════════════════════════════════════════
# LEXER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@test("Lexer: tokenizes basic function", "lexer")
def test_lex_basic_fn():
    from dim_lexer import Lexer
    from dim_token import TokenType
    code = "fn hello():\n    return 42\n"
    tokens = Lexer(code, "t.dim").tokenize()
    kinds  = [t.kind for t in tokens]
    assert TokenType.KEYWORD   in kinds
    assert TokenType.IDENTIFIER in kinds
    assert TokenType.INTEGER   in kinds


@test("Lexer: emits INDENT and DEDENT for blocks", "lexer")
def test_lex_indent_dedent():
    from dim_lexer import Lexer
    from dim_token import TokenType
    code = "fn f():\n    let x = 1\n"
    tokens = Lexer(code, "t.dim").tokenize()
    kinds  = [t.kind for t in tokens]
    assert TokenType.INDENT in kinds, "Expected INDENT"
    assert TokenType.DEDENT in kinds, "Expected DEDENT"


@test("Lexer: handles string escape sequences", "lexer")
def test_lex_string_escape():
    from dim_lexer import Lexer
    from dim_token import TokenType
    code = 'let s = "hello\\nworld"\n'
    tokens  = Lexer(code, "t.dim").tokenize()
    strtoks = [t for t in tokens if t.kind == TokenType.STRING]
    assert strtoks, "No STRING token found"
    assert "\n" in strtoks[0].value


@test("Lexer: tokenizes float literal", "lexer")
def test_lex_float():
    from dim_lexer import Lexer
    from dim_token import TokenType
    code = "let x = 3.14\n"
    tokens  = Lexer(code, "t.dim").tokenize()
    floattoks = [t for t in tokens if t.kind == TokenType.FLOAT]
    assert floattoks, "No FLOAT token"
    assert abs(floattoks[0].value - 3.14) < 1e-9


@test("Lexer: span line/col is correct", "lexer")
def test_lex_span():
    from dim_lexer import Lexer
    from dim_token import TokenType
    code = "fn foo():\n    let x = 42\n"
    tokens = Lexer(code, "t.dim").tokenize()
    fn_tok = next(t for t in tokens if t.kind == TokenType.KEYWORD and t.value == "fn")
    assert_eq(fn_tok.span.line_start, 1)
    assert_eq(fn_tok.span.col_start,  1)


# ═══════════════════════════════════════════════════════════════════════════════
# PARSER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@test("Parser: parses let binding", "parser")
def test_parse_let():
    from dim_ast import LetStmt, Literal
    code = "let x = 42\n"
    ast, diag = _parse(code)
    assert_no_errors(diag)
    assert len(ast.statements) == 1
    assert isinstance(ast.statements[0], LetStmt)
    stmt = ast.statements[0]
    assert_eq(stmt.name, "x")
    assert_eq(stmt.is_mut, False)
    assert isinstance(stmt.value, Literal)
    assert_eq(stmt.value.value, 42)


@test("Parser: parses mut let binding", "parser")
def test_parse_mut_let():
    from dim_ast import LetStmt
    ast, diag = _parse("let mut counter = 0\n")
    assert_no_errors(diag)
    assert isinstance(ast.statements[0], LetStmt)
    assert_eq(ast.statements[0].is_mut, True)


@test("Parser: parses function with params and return type", "parser")
def test_parse_function():
    from dim_ast import FunctionDef
    code = "fn add(x: i32, y: i32) -> i32:\n    return x\n"
    ast, diag = _parse(code)
    assert_no_errors(diag)
    fn = ast.statements[0]
    assert isinstance(fn, FunctionDef)
    assert_eq(fn.name, "add")
    assert_eq(len(fn.params), 2)
    assert_eq(fn.params[0].name, "x")
    assert_eq(fn.return_type.__repr__(), "i32")


@test("Parser: parses if/else", "parser")
def test_parse_if_else():
    from dim_ast import FunctionDef, IfStmt
    code = (
        "fn f():\n"
        "    if x > 0:\n"
        "        return 1\n"
        "    else:\n"
        "        return 0\n"
    )
    ast, diag = _parse(code)
    assert_no_errors(diag)
    fn   = ast.statements[0]
    stmt = fn.body[0]
    assert isinstance(stmt, IfStmt)
    assert stmt.else_branch is not None


@test("Parser: parses prompt definition", "parser")
def test_parse_prompt():
    from dim_ast import PromptDef
    code = (
        "prompt Classify:\n"
        "    role system: \"You are a classifier.\"\n"
        "    role user: \"Classify: this\"\n"
    )
    ast, diag = _parse(code)
    assert_no_errors(diag)
    p = ast.statements[0]
    assert isinstance(p, PromptDef)
    assert_eq(p.name, "Classify")
    assert_eq(len(p.roles), 2)


@test("Parser: parses struct definition", "parser")
def test_parse_struct():
    from dim_ast import StructDef
    code = "struct Point:\n    x: i32\n    y: i32\n"
    ast, diag = _parse(code)
    assert_no_errors(diag)
    s = ast.statements[0]
    assert isinstance(s, StructDef)
    assert_eq(s.name, "Point")
    assert_eq(len(s.fields), 2)


@test("Parser: parses enum definition", "parser")
def test_parse_enum():
    from dim_ast import EnumDef
    code = "enum Color:\n    Red\n    Green\n    Blue\n"
    ast, diag = _parse(code)
    assert_no_errors(diag)
    e = ast.statements[0]
    assert isinstance(e, EnumDef)
    assert_eq(e.name, "Color")
    assert_eq(len(e.variants), 3)


@test("Parser: parses binary expression with correct precedence", "parser")
def test_parse_expr_prec():
    from dim_ast import LetStmt, BinaryOp
    code = "let z = 2 + 3 * 4\n"
    ast, diag = _parse(code)
    assert_no_errors(diag)
    stmt = ast.statements[0]
    assert isinstance(stmt, LetStmt)
    # Should parse as 2 + (3 * 4), i.e. top-level op is "+"
    expr = stmt.value
    assert isinstance(expr, BinaryOp)
    assert_eq(expr.op, "+")
    assert isinstance(expr.right, BinaryOp)
    assert_eq(expr.right.op, "*")


@test("Parser: parses async function", "parser")
def test_parse_async_fn():
    from dim_ast import FunctionDef, AwaitExpr
    code = "async fn fetch():\n    let r = await get_data()\n"
    ast, diag = _parse(code)
    assert_no_errors(diag)
    fn = ast.statements[0]
    assert isinstance(fn, FunctionDef)
    assert_true(fn.is_async, "Expected is_async = True")


@test("Parser: parses borrow expression", "parser")
def test_parse_borrow():
    from dim_ast import LetStmt, BorrowExpr
    code = "let r = &x\n"
    ast, diag = _parse(code)
    assert_no_errors(diag)
    stmt = ast.statements[0]
    assert isinstance(stmt.value, BorrowExpr)
    assert_eq(stmt.value.mutable, False)


@test("Parser: parses mutable borrow", "parser")
def test_parse_mut_borrow():
    from dim_ast import LetStmt, BorrowExpr
    code = "let r = &mut x\n"
    ast, diag = _parse(code)
    stmt = ast.statements[0]
    assert isinstance(stmt.value, BorrowExpr)
    assert_eq(stmt.value.mutable, True)


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE CHECKER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@test("TypeChecker: literal types resolve correctly", "typecheck")
def test_tc_literal_types():
    from dim_types import I32, F32, BOOL, STR
    code = (
        "fn main():\n"
        "    let a = 42\n"
        "    let b = 3.14\n"
        "    let c = true\n"
        "    let d = \"hello\"\n"
    )
    ast, diag, _ = _type_check(code)
    assert_no_errors(diag)
    fn = ast.statements[0]
    let_a, let_b, let_c, let_d = fn.body
    assert_eq(repr(let_a.value.resolved_type), "i32")
    assert_eq(repr(let_b.value.resolved_type), "f32")
    assert_eq(repr(let_c.value.resolved_type), "bool")
    assert_eq(repr(let_d.value.resolved_type), "str")


@test("TypeChecker: undefined variable error", "typecheck")
def test_tc_undefined_var():
    code = "fn main():\n    let x = unknown_var\n"
    ast, diag, ok = _type_check(code)
    assert_true(not ok, "Should fail")
    assert_has_error(diag, "E0020")


@test("TypeChecker: type mismatch in binary op", "typecheck")
def test_tc_type_mismatch():
    code = "fn main():\n    let x = 1 + \"hello\"\n"
    ast, diag, ok = _type_check(code)
    assert_true(diag.has_errors, "Should report type mismatch")


@test("TypeChecker: function call arg count mismatch", "typecheck")
def test_tc_arg_count():
    code = (
        "fn add(x: i32, y: i32) -> i32:\n"
        "    return x\n"
        "fn main():\n"
        "    let r = add(1)\n"
    )
    ast, diag, ok = _type_check(code)
    assert_has_error(diag, "E0032")


@test("TypeChecker: immutable binding reassignment error", "typecheck")
def test_tc_immutable_assign():
    code = (
        "fn main():\n"
        "    let x = 5\n"
        "    x = 10\n"
    )
    ast, diag, ok = _type_check(code)
    assert_has_error(diag, "E0044")


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@test("Types: unification of identical primitives", "types")
def test_type_unify_prim():
    from dim_types import I32
    result = I32.unify(I32)
    assert_eq(repr(result), "i32")


@test("Types: type variable resolves on unification", "types")
def test_type_var_unify():
    from dim_types import TypeVar, I32
    tv = TypeVar("T")
    result = tv.unify(I32)
    assert_eq(repr(tv.root()), "i32")


@test("Types: numeric promotion float wins", "types")
def test_numeric_promo_float():
    from dim_types import numeric_promotion, I32, F64, PrimType, PrimKind
    f64 = PrimType(PrimKind.F64)
    result = numeric_promotion(I32, f64)
    assert_eq(repr(result), "f64")


@test("Types: RefType repr", "types")
def test_ref_type_repr():
    from dim_types import RefType, I32
    r = RefType(I32, mutable=True)
    assert_eq(repr(r), "&mut i32")


@test("Types: PromptType repr", "types")
def test_prompt_type_repr():
    from dim_types import PromptType, STR, I32
    p = PromptType(STR, I32, deterministic=True)
    assert "Prompt" in repr(p)
    assert "deterministic" in repr(p)


# ═══════════════════════════════════════════════════════════════════════════════
# MIR LOWERING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@test("MIR: simple function lowers to MIRFunction", "mir")
def test_mir_simple_fn():
    code = "fn add(x: i32, y: i32) -> i32:\n    return x\n"
    module, diag = _lower(code)
    assert_true(len(module.functions) > 0, "Expected at least 1 MIR function")
    fn = module.functions[0]
    assert_eq(fn.name, "add")
    assert_eq(len(fn.params), 2)


@test("MIR: if statement creates branch terminator", "mir")
def test_mir_if_branch():
    from dim_mir import Branch
    code = (
        "fn classify(x: i32) -> i32:\n"
        "    if x > 0:\n"
        "        return 1\n"
        "    else:\n"
        "        return 0\n"
    )
    module, diag = _lower(code)
    fn = module.functions[0]
    terminators = [bb.terminator for bb in fn.blocks]
    has_branch  = any(isinstance(t, Branch) for t in terminators)
    assert_true(has_branch, "Expected Branch terminator in MIR")


@test("MIR: liveness analysis produces live sets", "mir")
def test_mir_liveness():
    from dim_mir import cfg_liveness
    code = "fn f(x: i32) -> i32:\n    let y = x\n    return y\n"
    module, _ = _lower(code)
    fn = module.functions[0]
    live_in, live_out = cfg_liveness(fn)
    assert isinstance(live_in, dict)
    assert isinstance(live_out, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# BORROW CHECKER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@test("BorrowChecker: double mutable borrow detected", "borrow")
def test_borrow_double_mut():
    """
    Verify the borrow checker catches two simultaneous mutable borrows
    at the MIR level by directly constructing a test MIR.
    """
    from dim_types import I32
    from dim_mir import (
        Local, Place, Mutability, BorrowKind,
        BasicBlock, MIRFunction,
        StorageLive, Borrow,
        Return, ConstOperand,
    )
    from dim_borrow_checker import BorrowChecker
    from dim_diagnostic import DiagnosticBag

    x = Local(0, I32, Mutability.Mut, "x")
    r1 = Local(1, I32, Mutability.Not, "r1")
    r2 = Local(2, I32, Mutability.Not, "r2")

    bb = BasicBlock(0)
    bb.stmts = [
        StorageLive(x),
        Borrow(r1, BorrowKind.Mutable, Place(x)),
        Borrow(r2, BorrowKind.Mutable, Place(x)),  # ERROR: double mutable
    ]
    bb.terminator = Return(ConstOperand(None, I32))

    fn = MIRFunction("test", [x], I32, {0: x, 1: r1, 2: r2}, [bb])
    diag = DiagnosticBag()
    checker = BorrowChecker(fn, diag)
    checker.check()
    assert_has_error(diag, "E0041")


@test("BorrowChecker: use after move detected", "borrow")
def test_borrow_use_after_move():
    """
    Verify the borrow checker catches use of a moved value.
    """
    from dim_types import I32
    from dim_mir import (
        Local, Place, Mutability,
        BasicBlock, MIRFunction,
        StorageLive, StorageDead, Assign,
        Return, PlaceOperand, UseRValue, ConstOperand,
    )
    from dim_borrow_checker import BorrowChecker
    from dim_diagnostic import DiagnosticBag

    x   = Local(0, I32, Mutability.Not, "x")
    y   = Local(1, I32, Mutability.Not, "y")

    bb = BasicBlock(0)
    bb.stmts = [
        StorageLive(x),
        Assign(Place(y), UseRValue(PlaceOperand(Place(x)))),
        StorageDead(x),   # x is now dropped
        # Try to use x after it's dead
        Assign(Place(y), UseRValue(PlaceOperand(Place(x)))),  # ERROR
    ]
    bb.terminator = Return(ConstOperand(None, I32))

    fn = MIRFunction("test_uam", [x], I32, {0: x, 1: y}, [bb])
    diag = DiagnosticBag()
    checker = BorrowChecker(fn, diag)
    checker.check()
    assert_has_error(diag, "E0040")


@test("BorrowChecker: immutable binding mutation detected", "borrow")
def test_borrow_immutable_assign():
    from dim_types import I32
    from dim_mir import (
        Local, Place, Mutability,
        BasicBlock, MIRFunction,
        StorageLive, Assign,
        Return, ConstOperand, UseRValue,
    )
    from dim_borrow_checker import BorrowChecker
    from dim_diagnostic import DiagnosticBag

    x = Local(0, I32, Mutability.Not, "x")
    bb = BasicBlock(0)
    bb.stmts = [
        StorageLive(x),
        Assign(Place(x), UseRValue(ConstOperand(5, I32))),
        Assign(Place(x), UseRValue(ConstOperand(10, I32))),  # ERROR: immutable
    ]
    bb.terminator = Return(ConstOperand(None, I32))

    fn = MIRFunction("test_imm", [x], I32, {0: x}, [bb])
    diag = DiagnosticBag()
    checker = BorrowChecker(fn, diag)
    checker.check()
    assert_has_error(diag, "E0044")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else None
    ok  = run_tests(tag)
    sys.exit(0 if ok else 1)
