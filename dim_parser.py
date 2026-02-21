# dim_parser.py — Production Parser for Dim (v0.2)
#
# Recursive descent parser using structured Token objects (from dim_lexer.py).
# Produces a span-annotated AST (from dim_ast.py v0.2).
# Reports errors through DiagnosticBag.

from __future__ import annotations
from typing import List, Optional, Tuple
from dim_token import Token, TokenType, Span
from dim_diagnostic import DiagnosticBag
from dim_types import (
    Type, resolve_builtin, RefType, GenericType, TensorType,
    PrimType, PrimKind, UnknownType, F32
)
from dim_ast import *


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: List[Token], source: str = "",
                 filename: str = "<stdin>"):
        self.tokens   = tokens
        self.pos      = 0
        self.source   = source
        self.filename = filename
        self.diag     = DiagnosticBag(source, filename)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        if idx >= len(self.tokens):
            return self.tokens[-1]  # EOF token
        return self.tokens[idx]

    def _advance(self) -> Token:
        tok = self._peek()
        if tok.kind != TokenType.EOF:
            self.pos += 1
        return tok

    def _check(self, kind: TokenType) -> bool:
        return self._peek().kind == kind

    def _check_kw(self, *words: str) -> bool:
        t = self._peek()
        return t.kind == TokenType.KEYWORD and t.value in words

    def _match(self, kind: TokenType) -> Optional[Token]:
        if self._check(kind):
            return self._advance()
        return None

    def _match_kw(self, *words: str) -> Optional[Token]:
        if self._check_kw(*words):
            return self._advance()
        return None

    def _expect(self, kind: TokenType, msg: str = "") -> Token:
        tok = self._peek()
        if tok.kind != kind:
            self.diag.error("E0010",
                msg or f"Expected {kind.name}, found `{tok.value}`",
                tok.span)
            # Error recovery: don't advance, return a synthetic token
            return Token(kind, None, tok.span)
        return self._advance()

    def _expect_kw(self, word: str) -> Token:
        tok = self._peek()
        if not (tok.kind == TokenType.KEYWORD and tok.value == word):
            self.diag.error("E0010",
                f"Expected keyword `{word}`, found `{tok.value}`",
                tok.span)
            return Token(TokenType.KEYWORD, word, tok.span)
        return self._advance()

    def _skip_newlines(self):
        while self._check(TokenType.NEWLINE):
            self._advance()

    def _start_span(self) -> Span:
        return self._peek().span

    def _end_span(self, start: Span) -> Span:
        return start.merge(self._peek(-1 if self.pos > 0 else 0).span
                           if self.pos > 0 else start)

    # ── Programme ─────────────────────────────────────────────────────────────

    def parse_program(self) -> Program:
        start  = self._start_span()
        stmts: List[Statement] = []
        self._skip_newlines()
        while not self._check(TokenType.EOF):
            s = self._parse_top_level()
            if s:
                stmts.append(s)
            self._skip_newlines()
        return Program(stmts, span=self._end_span(start))

    def _parse_top_level(self) -> Optional[Statement]:
        self._skip_newlines()
        if self._check_kw("fn") or self._check_kw("async"): return self._parse_function()
        if self._check_kw("struct"): return self._parse_struct()
        if self._check_kw("enum"):   return self._parse_enum()
        if self._check_kw("trait"):  return self._parse_trait()
        if self._check_kw("impl"):   return self._parse_impl()
        if self._check_kw("prompt"): return self._parse_prompt()
        if self._check_kw("actor"):  return self._parse_actor()
        if self._check_kw("import"): return self._parse_import()
        if self._check(TokenType.NEWLINE):
            self._advance()
            return None
        return self._parse_statement()

    # ── Blocks ────────────────────────────────────────────────────────────────

    def _parse_block(self) -> List[Statement]:
        """Parse a colon-prefixed indented block."""
        self._expect(TokenType.COLON, "Expected `:` before block")
        self._skip_newlines()
        self._expect(TokenType.INDENT, "Expected indented block")
        body: List[Statement] = []
        while not self._check(TokenType.DEDENT) and not self._check(TokenType.EOF):
            self._skip_newlines()
            if self._check(TokenType.DEDENT):
                break
            s = self._parse_statement()
            if s:
                body.append(s)
        self._match(TokenType.DEDENT)
        return body

    # ── Statements ────────────────────────────────────────────────────────────

    def _parse_statement(self) -> Optional[Statement]:
        self._skip_newlines()
        start = self._start_span()

        if self._check_kw("let"):    return self._parse_let()
        if self._check_kw("return"): return self._parse_return()
        if self._check_kw("if"):     return self._parse_if()
        if self._check_kw("while"):  return self._parse_while()
        if self._check_kw("for"):    return self._parse_for()
        if self._check_kw("match"):  return self._parse_match()
        if self._check_kw("break"):
            self._advance()
            self._skip_newlines()
            return BreakStmt(span=start)
        if self._check_kw("continue"):
            self._advance()
            self._skip_newlines()
            return ContinueStmt(span=start)
        if self._check_kw("unsafe"):
            self._advance()
            body = self._parse_block()
            return UnsafeBlock(body, span=self._end_span(start))

        # Expression statement or assignment
        expr = self._parse_expression()
        # Check for assignment
        if self._check(TokenType.EQ):
            self._advance()
            value = self._parse_expression()
            self._skip_newlines()
            return AssignStmt(expr, "=", value, span=self._end_span(start))
        self._skip_newlines()
        return ExprStmt(expr, span=self._end_span(start))

    def _parse_let(self) -> LetStmt:
        start = self._start_span()
        self._expect_kw("let")
        is_mut = bool(self._match_kw("mut"))
        name   = self._expect(TokenType.IDENTIFIER).value
        ann    = None
        if self._match(TokenType.COLON):
            ann = self._parse_type()
        self._expect(TokenType.EQ, "Expected `=` in let binding")
        value  = self._parse_expression()
        self._skip_newlines()
        return LetStmt(name, is_mut, ann, value, span=self._end_span(start))

    def _parse_return(self) -> ReturnStmt:
        start = self._start_span()
        self._expect_kw("return")
        value = None
        if not self._check(TokenType.NEWLINE) and not self._check(TokenType.EOF):
            value = self._parse_expression()
        self._skip_newlines()
        return ReturnStmt(value, span=self._end_span(start))

    def _parse_if(self) -> IfStmt:
        start = self._start_span()
        self._expect_kw("if")
        cond      = self._parse_expression()
        then_body = self._parse_block()
        elifs: List[Tuple] = []
        else_body = None
        while self._check_kw("elif"):
            self._advance()
            elif_cond = self._parse_expression()
            elif_body = self._parse_block()
            elifs.append((elif_cond, elif_body))
        if self._check_kw("else"):
            self._advance()
            else_body = self._parse_block()
        return IfStmt(cond, then_body, elifs, else_body, span=self._end_span(start))

    def _parse_while(self) -> WhileStmt:
        start = self._start_span()
        self._expect_kw("while")
        cond = self._parse_expression()
        body = self._parse_block()
        return WhileStmt(cond, body, span=self._end_span(start))

    def _parse_for(self) -> ForStmt:
        start = self._start_span()
        self._expect_kw("for")
        iterator = self._expect(TokenType.IDENTIFIER).value
        self._expect_kw("in")
        iterable = self._parse_expression()
        body     = self._parse_block()
        return ForStmt(iterator, iterable, body, span=self._end_span(start))

    def _parse_match(self) -> MatchStmt:
        start = self._start_span()
        self._expect_kw("match")
        expr = self._parse_expression()
        self._expect(TokenType.COLON)
        self._skip_newlines()
        self._expect(TokenType.INDENT)
        arms: List[MatchArm] = []
        while not self._check(TokenType.DEDENT) and not self._check(TokenType.EOF):
            self._skip_newlines()
            if self._check(TokenType.DEDENT):
                break
            arm_start = self._start_span()
            pattern   = self._parse_expression()
            guard     = None
            if self._check_kw("if"):
                self._advance()
                guard = self._parse_expression()
            body = self._parse_block()
            arms.append(MatchArm(pattern, guard, body, span=self._end_span(arm_start)))
        self._match(TokenType.DEDENT)
        return MatchStmt(expr, arms, span=self._end_span(start))

    # ── Top-level declarations ────────────────────────────────────────────────

    def _parse_function(self) -> FunctionDef:
        start   = self._start_span()
        is_async = False
        if self._check_kw("async"):
            is_async = True
            self._advance()
        self._expect_kw("fn")
        name = self._expect(TokenType.IDENTIFIER).value
        # Optional generics: fn foo[T, U](...)
        generics: List[str] = []
        if self._check(TokenType.LBRACKET):
            self._advance()
            while not self._check(TokenType.RBRACKET):
                generics.append(self._expect(TokenType.IDENTIFIER).value)
                if not self._match(TokenType.COMMA):
                    break
            self._expect(TokenType.RBRACKET)
        self._expect(TokenType.LPAREN)
        params = self._parse_params()
        self._expect(TokenType.RPAREN)
        ret = None
        if self._check(TokenType.ARROW):
            self._advance()
            ret = self._parse_type()
        body = self._parse_block()
        return FunctionDef(name, params, ret, body, is_async,
                           span=self._end_span(start), generics=generics)

    def _parse_params(self) -> List[Param]:
        params: List[Param] = []
        while not self._check(TokenType.RPAREN) and not self._check(TokenType.EOF):
            p_start = self._start_span()
            is_mut  = bool(self._match_kw("mut"))
            name    = self._expect(TokenType.IDENTIFIER).value
            self._expect(TokenType.COLON)
            ty      = self._parse_type()
            params.append(Param(name, ty, is_mut, span=self._end_span(p_start)))
            if not self._match(TokenType.COMMA):
                break
        return params

    def _parse_struct(self) -> StructDef:
        start = self._start_span()
        self._expect_kw("struct")
        name     = self._expect(TokenType.IDENTIFIER).value
        generics = self._parse_generic_params()
        self._expect(TokenType.COLON)
        self._skip_newlines()
        self._expect(TokenType.INDENT)
        fields: List[Tuple] = []
        while not self._check(TokenType.DEDENT) and not self._check(TokenType.EOF):
            self._skip_newlines()
            if self._check(TokenType.DEDENT):
                break
            fname = self._expect(TokenType.IDENTIFIER).value
            self._expect(TokenType.COLON)
            ftype = self._parse_type()
            fields.append((fname, ftype, False))
            self._skip_newlines()
        self._match(TokenType.DEDENT)
        return StructDef(name, fields, generics, span=self._end_span(start))

    def _parse_enum(self) -> EnumDef:
        start = self._start_span()
        self._expect_kw("enum")
        name     = self._expect(TokenType.IDENTIFIER).value
        generics = self._parse_generic_params()
        self._expect(TokenType.COLON)
        self._skip_newlines()
        self._expect(TokenType.INDENT)
        variants: List[Tuple] = []
        while not self._check(TokenType.DEDENT) and not self._check(TokenType.EOF):
            self._skip_newlines()
            if self._check(TokenType.DEDENT):
                break
            vname = self._expect(TokenType.IDENTIFIER).value
            vtypes = []
            if self._check(TokenType.LPAREN):
                self._advance()
                while not self._check(TokenType.RPAREN):
                    vtypes.append(self._parse_type())
                    if not self._match(TokenType.COMMA):
                        break
                self._expect(TokenType.RPAREN)
            variants.append((vname, vtypes or None))
            self._skip_newlines()
        self._match(TokenType.DEDENT)
        return EnumDef(name, variants, generics, span=self._end_span(start))

    def _parse_trait(self) -> TraitDef:
        start = self._start_span()
        self._expect_kw("trait")
        name     = self._expect(TokenType.IDENTIFIER).value
        generics = self._parse_generic_params()
        self._expect(TokenType.COLON)
        self._skip_newlines()
        self._expect(TokenType.INDENT)
        methods: List[FunctionDef] = []
        while not self._check(TokenType.DEDENT) and not self._check(TokenType.EOF):
            self._skip_newlines()
            if self._check(TokenType.DEDENT):
                break
            if self._check_kw("fn") or self._check_kw("async"):
                methods.append(self._parse_function())
            else:
                self._advance()  # error recovery
        self._match(TokenType.DEDENT)
        return TraitDef(name, methods, generics, span=self._end_span(start))

    def _parse_impl(self) -> ImplBlock:
        start    = self._start_span()
        self._expect_kw("impl")
        generics = self._parse_generic_params()
        name     = self._expect(TokenType.IDENTIFIER).value
        trait_nm = None
        if self._check_kw("for"):
            trait_nm = name
            self._advance()
            name = self._expect(TokenType.IDENTIFIER).value
        body = self._parse_block()
        methods = [s for s in body if isinstance(s, FunctionDef)]
        return ImplBlock(name, trait_nm, methods, generics, span=self._end_span(start))

    def _parse_prompt(self) -> PromptDef:
        start = self._start_span()
        self._expect_kw("prompt")
        name = self._expect(TokenType.IDENTIFIER).value
        base = None
        if self._check_kw("extends"):
            self._advance()
            base = self._expect(TokenType.IDENTIFIER).value
        self._expect(TokenType.COLON)
        self._skip_newlines()
        self._expect(TokenType.INDENT)
        roles:      List[PromptRole] = []
        output_type = None
        while not self._check(TokenType.DEDENT) and not self._check(TokenType.EOF):
            self._skip_newlines()
            if self._check(TokenType.DEDENT):
                break
            if self._check_kw("role"):
                r_start = self._start_span()
                self._advance()
                role_name = self._expect(TokenType.IDENTIFIER).value
                self._expect(TokenType.COLON)
                content   = self._expect(TokenType.STRING).value or ""
                roles.append(PromptRole(role_name, content, span=self._end_span(r_start)))
            elif self._check_kw("output"):
                self._advance()
                self._expect(TokenType.COLON)
                output_type = self._parse_type()
            self._skip_newlines()
        self._match(TokenType.DEDENT)
        return PromptDef(name, base, roles, None, output_type,
                         span=self._end_span(start))

    def _parse_actor(self) -> ActorDef:
        start = self._start_span()
        self._expect_kw("actor")
        name = self._expect(TokenType.IDENTIFIER).value
        self._expect(TokenType.COLON)
        self._skip_newlines()
        self._expect(TokenType.INDENT)
        fields: List[Tuple]       = []
        handlers: List[FunctionDef] = []
        while not self._check(TokenType.DEDENT) and not self._check(TokenType.EOF):
            self._skip_newlines()
            if self._check(TokenType.DEDENT):
                break
            if self._check_kw("receive"):
                self._advance()
                # 're-parse as a function'
                fn_tok_start = self._start_span()
                msg_name = self._expect(TokenType.IDENTIFIER).value
                self._expect(TokenType.LPAREN)
                params = self._parse_params()
                self._expect(TokenType.RPAREN)
                ret = None
                if self._check(TokenType.ARROW):
                    self._advance(); ret = self._parse_type()
                body = self._parse_block()
                handlers.append(FunctionDef(msg_name, params, ret, body,
                                             span=self._end_span(fn_tok_start)))
            else:
                fname = self._expect(TokenType.IDENTIFIER).value
                self._expect(TokenType.COLON)
                ftype = self._parse_type()
                default = None
                if self._check(TokenType.EQ):
                    self._advance()
                    default = self._parse_expression()
                fields.append((fname, ftype, default))
                self._skip_newlines()
        self._match(TokenType.DEDENT)
        return ActorDef(name, fields, handlers, span=self._end_span(start))

    def _parse_import(self) -> ImportStmt:
        start = self._start_span()
        self._expect_kw("import")
        path: List[str] = [self._expect(TokenType.IDENTIFIER).value]
        while self._check(TokenType.DOT):
            self._advance()
            path.append(self._expect(TokenType.IDENTIFIER).value)
        alias = None
        if self._check_kw("as"):
            self._advance()
            alias = self._expect(TokenType.IDENTIFIER).value
        self._skip_newlines()
        return ImportStmt(path, None, alias, span=self._end_span(start))

    def _parse_generic_params(self) -> List[str]:
        params: List[str] = []
        if self._check(TokenType.LBRACKET):
            self._advance()
            while not self._check(TokenType.RBRACKET):
                params.append(self._expect(TokenType.IDENTIFIER).value)
                if not self._match(TokenType.COMMA):
                    break
            self._expect(TokenType.RBRACKET)
        return params

    # ── Type parsing ──────────────────────────────────────────────────────────

    def _parse_type(self) -> Type:
        # &type or &mut type
        if self._check(TokenType.AMP):
            self._advance()
            is_mut = bool(self._match_kw("mut"))
            inner  = self._parse_type()
            return RefType(inner, is_mut)
        if not self._check(TokenType.IDENTIFIER):
            return UnknownType()
        name = self._advance().value
        # Generic: Name[T, U]
        if self._check(TokenType.LBRACKET):
            self._advance()
            args: List[Type] = []
            while not self._check(TokenType.RBRACKET):
                args.append(self._parse_type())
                if not self._match(TokenType.COMMA):
                    break
            self._expect(TokenType.RBRACKET)
            if name == "Tensor" and args:
                shape: List[Optional[int]] = []
                return TensorType(args[0] if args else PrimType(PrimKind.F32), shape)
            return GenericType(name, args)
        builtin = resolve_builtin(name)
        return builtin if builtin else GenericType(name, [])

    # ── Expressions (Pratt parser) ────────────────────────────────────────────

    _PREC: dict = {
        "or":  1,
        "and": 2,
        "==":  3, "!=": 3, "<": 3, ">": 3, "<=": 3, ">=": 3,
        "+":   4, "-":  4,
        "*":   5, "/":  5, "%": 5,
    }

    def _parse_expression(self) -> Expression:
        return self._parse_binary(0)

    def _parse_binary(self, min_prec: int) -> Expression:
        left = self._parse_unary()
        while True:
            tok = self._peek()
            op  = tok.value
            # Determine if current token is a binary operator
            prec = None
            if tok.kind == TokenType.KEYWORD and op in ("and", "or"):
                prec = self._PREC.get(op)
            elif tok.kind in (TokenType.PLUS, TokenType.MINUS,
                              TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
                prec = self._PREC.get(op)
            elif tok.kind in (TokenType.EQEQ, TokenType.NEQ,
                              TokenType.LT, TokenType.GT,
                              TokenType.LTE, TokenType.GTE):
                prec = self._PREC.get(op)
            if prec is None or prec <= min_prec:
                break
            self._advance()
            right = self._parse_binary(prec)
            start = left.span.merge(right.span)
            left  = BinaryOp(left, op, right, span=start)
        return left

    def _parse_unary(self) -> Expression:
        start = self._start_span()
        if self._check_kw("not") or self._check(TokenType.NOT):
            op = self._advance().value
            operand = self._parse_unary()
            return UnaryOp(op, operand, span=self._end_span(start))
        if self._check(TokenType.MINUS):
            op = self._advance().value
            operand = self._parse_unary()
            return UnaryOp(op, operand, span=self._end_span(start))
        if self._check(TokenType.AMP):
            self._advance()
            is_mut = bool(self._match_kw("mut"))
            operand = self._parse_unary()
            return BorrowExpr(operand, is_mut, span=self._end_span(start))
        if self._check(TokenType.STAR):
            self._advance()
            operand = self._parse_unary()
            return DerefExpr(operand, span=self._end_span(start))
        if self._check_kw("await"):
            self._advance()
            operand = self._parse_unary()
            return AwaitExpr(operand, span=self._end_span(start))
        return self._parse_postfix()

    def _parse_postfix(self) -> Expression:
        expr = self._parse_primary()
        while True:
            if self._check(TokenType.LPAREN):
                # Function call
                self._advance()
                args: List[Expression] = []
                while not self._check(TokenType.RPAREN) and not self._check(TokenType.EOF):
                    args.append(self._parse_expression())
                    if not self._match(TokenType.COMMA):
                        break
                self._expect(TokenType.RPAREN)
                expr = Call(expr, args, span=expr.span)
            elif self._check(TokenType.DOT):
                self._advance()
                member = self._expect(TokenType.IDENTIFIER).value
                if self._check(TokenType.LPAREN):
                    self._advance()
                    args = []
                    while not self._check(TokenType.RPAREN) and not self._check(TokenType.EOF):
                        args.append(self._parse_expression())
                        if not self._match(TokenType.COMMA):
                            break
                    self._expect(TokenType.RPAREN)
                    expr = MethodCall(expr, member, args, span=expr.span)
                else:
                    expr = MemberAccess(expr, member, span=expr.span)
            elif self._check(TokenType.LBRACKET):
                self._advance()
                index = self._parse_expression()
                self._expect(TokenType.RBRACKET)
                expr  = IndexAccess(expr, index, span=expr.span)
            else:
                break
        return expr

    def _parse_primary(self) -> Expression:
        start = self._start_span()
        tok   = self._peek()

        if tok.kind == TokenType.INTEGER:
            self._advance()
            return Literal(tok.value, span=tok.span)
        if tok.kind == TokenType.FLOAT:
            self._advance()
            return Literal(tok.value, span=tok.span)
        if tok.kind == TokenType.STRING:
            self._advance()
            return Literal(tok.value, span=tok.span)
        if tok.kind == TokenType.BOOL:
            self._advance()
            return Literal(tok.value, span=tok.span)
        if tok.kind == TokenType.KEYWORD and tok.value in ("true", "false"):
            self._advance()
            return Literal(tok.value == "true", span=tok.span)
        if tok.kind == TokenType.KEYWORD and tok.value == "none":
            self._advance()
            return Literal(None, span=tok.span)

        if tok.kind == TokenType.IDENTIFIER:
            self._advance()
            return Identifier(tok.value, span=tok.span)

        if tok.kind == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expression()
            self._expect(TokenType.RPAREN)
            return expr

        if tok.kind == TokenType.LBRACKET:
            self._advance()
            elements: List[Expression] = []
            while not self._check(TokenType.RBRACKET) and not self._check(TokenType.EOF):
                elements.append(self._parse_expression())
                if not self._match(TokenType.COMMA):
                    break
            self._expect(TokenType.RBRACKET)
            return ListLiteral(elements, span=self._end_span(start))

        # Error recovery
        self.diag.error("E0011",
            f"Expected expression, found `{tok.value}`",
            tok.span)
        self._advance()
        return Literal(0, span=tok.span)
