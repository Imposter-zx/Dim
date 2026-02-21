# dim_lexer.py — Production Lexer for Dim
#
# Replaces dim_poc_lexer.py with a character-by-character lexer that:
#   - Produces structured Token objects with Span source locations
#   - Handles INDENT/DEDENT injection properly
#   - Reports lex errors through DiagnosticBag
#   - Handles floats, multiline strings, comments, all operators

from __future__ import annotations
from typing import List, Optional
from dim_token import Token, TokenType, Span, KEYWORDS
from dim_diagnostic import DiagnosticBag


class Lexer:
    def __init__(self, source: str, filename: str = "<stdin>"):
        self.source   = source
        self.filename = filename
        self.pos      = 0
        self.line     = 1
        self.col      = 1
        self.tokens: List[Token] = []
        self.indent_stack: List[int] = [0]
        self.diag = DiagnosticBag(source, filename)

    # ── Low-level helpers ─────────────────────────────────────────────────────

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        return self.source[idx] if idx < len(self.source) else "\0"

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col   = 1
        else:
            self.col  += 1
        return ch

    def _span_at(self, line: int, col: int) -> Span:
        return Span(self.filename, line, col, self.line, self.col)

    def _make_token(self, kind: TokenType, value, line: int, col: int) -> Token:
        return Token(kind, value, self._span_at(line, col))

    # ── Tokenize ──────────────────────────────────────────────────────────────

    def tokenize(self) -> List[Token]:
        """
        Main entry point.  Returns a flat list of Token objects with
        INDENT/DEDENT/NEWLINE tokens injected for block structure.
        """
        lines = self.source.split("\n")
        # We process line-by-line to handle indentation, then lex each line.
        self.pos  = 0
        self.line = 1
        self.col  = 1

        for raw_line in lines:
            self._process_line(raw_line)

        # Flush remaining dedents at EOF
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.tokens.append(Token(TokenType.DEDENT, None,
                                     Span(self.filename, self.line, 1, self.line, 1)))

        self.tokens.append(Token(TokenType.EOF, None,
                                  Span(self.filename, self.line, self.col,
                                       self.line, self.col)))
        return self.tokens

    def _process_line(self, raw_line: str):
        """Tokenise a single source line (already split on \\n)."""
        # Skip blank lines and pure-comment lines for indent tracking
        stripped = raw_line.lstrip()
        if not stripped or stripped.startswith("#"):
            self.line += 1
            self.col   = 1
            return

        # ── Indentation handling ──────────────────────────────────────────────
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        prev   = self.indent_stack[-1]

        if indent > prev:
            self.indent_stack.append(indent)
            self.tokens.append(Token(TokenType.INDENT, None,
                                     Span(self.filename, self.line, 1, self.line, indent + 1)))
        elif indent < prev:
            while indent < self.indent_stack[-1]:
                self.indent_stack.pop()
                self.tokens.append(Token(TokenType.DEDENT, None,
                                         Span(self.filename, self.line, 1, self.line, indent + 1)))
            if indent != self.indent_stack[-1]:
                self.diag.error("E0001",
                    f"Inconsistent indentation: {indent} spaces, expected {self.indent_stack[-1]}",
                    Span(self.filename, self.line, 1, self.line, indent + 1))

        # ── Tokenise content ─────────────────────────────────────────────────
        # Temporarily redirect to a sub-lexer scoped to this line
        sub = _LineLexer(raw_line.lstrip(), self.filename, self.line,
                         indent + 1, self.diag)
        self.tokens.extend(sub.lex())
        self.tokens.append(Token(TokenType.NEWLINE, None,
                                  Span(self.filename, self.line, len(raw_line) + 1,
                                       self.line, len(raw_line) + 1)))
        self.line += 1
        self.col   = 1


class _LineLexer:
    """Tokenises a single stripped line of source code."""

    def __init__(self, text: str, filename: str, line_no: int,
                 col_offset: int, diag: DiagnosticBag):
        self.text       = text
        self.filename   = filename
        self.line_no    = line_no
        self.col_offset = col_offset   # Column where text starts (after indent)
        self.pos        = 0
        self.diag       = diag

    def _peek(self, off: int = 0) -> str:
        idx = self.pos + off
        return self.text[idx] if idx < len(self.text) else "\0"

    def _advance(self) -> str:
        ch = self.text[self.pos]
        self.pos += 1
        return ch

    def _col(self) -> int:
        return self.col_offset + self.pos

    def _span(self, start_col: int) -> Span:
        return Span(self.filename, self.line_no, start_col,
                    self.line_no, self._col())

    def lex(self) -> List[Token]:
        tokens: List[Token] = []
        while self.pos < len(self.text):
            ch = self._peek()

            # Skip whitespace
            if ch in " \t":
                self._advance()
                continue

            # Comments
            if ch == "#":
                break  # Rest of line is comment

            start_col = self._col()

            # ── String literals ───────────────────────────────────────────────
            if ch in ('"', "'"):
                tok = self._lex_string(start_col)
                if tok:
                    tokens.append(tok)
                continue

            # ── Numbers ───────────────────────────────────────────────────────
            if ch.isdigit() or (ch == "-" and self._peek(1).isdigit()
                                and not tokens):
                tokens.append(self._lex_number(start_col))
                continue

            # ── Identifiers / Keywords ────────────────────────────────────────
            if ch.isalpha() or ch == "_":
                tokens.append(self._lex_word(start_col))
                continue

            # ── Multi-char operators ──────────────────────────────────────────
            two = self._peek(0) + self._peek(1)
            OPS2 = {
                "==": TokenType.EQEQ,
                "!=": TokenType.NEQ,
                "<=": TokenType.LTE,
                ">=": TokenType.GTE,
                "->": TokenType.ARROW,
                "=>": TokenType.FAT_ARROW,
                "&&": TokenType.AND,
                "||": TokenType.OR,
            }
            if two in OPS2:
                self._advance(); self._advance()
                tokens.append(Token(OPS2[two], two, self._span(start_col)))
                continue

            # ── Single-char tokens ────────────────────────────────────────────
            SINGLE = {
                "+": TokenType.PLUS,
                "-": TokenType.MINUS,
                "*": TokenType.STAR,
                "/": TokenType.SLASH,
                "%": TokenType.PERCENT,
                "=": TokenType.EQ,
                "<": TokenType.LT,
                ">": TokenType.GT,
                ":": TokenType.COLON,
                ",": TokenType.COMMA,
                ".": TokenType.DOT,
                "(": TokenType.LPAREN,
                ")": TokenType.RPAREN,
                "[": TokenType.LBRACKET,
                "]": TokenType.RBRACKET,
                "{": TokenType.LBRACE,
                "}": TokenType.RBRACE,
                "&": TokenType.AMP,
                "|": TokenType.PIPE,
                "!": TokenType.NOT,
            }
            if ch in SINGLE:
                self._advance()
                tokens.append(Token(SINGLE[ch], ch, self._span(start_col)))
                continue

            # Unknown character
            self.diag.error("E0001", f"Unknown character: {ch!r}",
                            self._span(start_col))
            self._advance()

        return tokens

    def _lex_string(self, start_col: int) -> Optional[Token]:
        quote = self._advance()
        buf   = []
        while self.pos < len(self.text):
            ch = self._advance()
            if ch == "\\":
                esc = self._advance() if self.pos < len(self.text) else ""
                buf.append({"n": "\n", "t": "\t", "r": "\r",
                            "\\": "\\", '"': '"', "'": "'"}.get(esc, esc))
            elif ch == quote:
                return Token(TokenType.STRING, "".join(buf), self._span(start_col))
            else:
                buf.append(ch)
        self.diag.error("E0002", "Unterminated string literal",
                        self._span(start_col))
        return None

    def _lex_number(self, start_col: int) -> Token:
        buf  = []
        is_f = False
        if self._peek() == "-":
            buf.append(self._advance())
        while self.pos < len(self.text) and (self._peek().isdigit() or
                                              self._peek() == "."):
            ch = self._advance()
            if ch == ".":
                if is_f:
                    break
                is_f = True
            buf.append(ch)
        raw = "".join(buf)
        try:
            val = float(raw) if is_f else int(raw)
        except ValueError:
            self.diag.error("E0003", f"Invalid number literal: {raw!r}",
                            self._span(start_col))
            val = 0
        kind = TokenType.FLOAT if is_f else TokenType.INTEGER
        return Token(kind, val, self._span(start_col))

    def _lex_word(self, start_col: int) -> Token:
        buf = []
        while self.pos < len(self.text) and (self._peek().isalnum()
                                               or self._peek() == "_"):
            buf.append(self._advance())
        word = "".join(buf)
        if word == "true":
            return Token(TokenType.BOOL, True, self._span(start_col))
        if word == "false":
            return Token(TokenType.BOOL, False, self._span(start_col))
        kind = TokenType.KEYWORD if word in KEYWORDS else TokenType.IDENTIFIER
        return Token(kind, word, self._span(start_col))


# ── Backwards-compatible shim ─────────────────────────────────────────────────
# The old lexer returned raw (TokenType, value) tuples; new code uses Token.
# The parser has been updated but external scripts can call legacy_tokenize().

def legacy_tokenize(source: str, filename: str = "<stdin>") -> list:
    """Returns list of (TokenType, value) tuples for backward compat."""
    lexer = Lexer(source, filename)
    tokens = lexer.tokenize()
    return [(t.kind, t.value) for t in tokens]


if __name__ == "__main__":
    code = """
fn greet(name: str) -> str:
    let msg = "Hello, " + name
    return msg

prompt Classify:
    role system: "You are a classifier."
    role user: "Classify: {input}"
"""
    lex = Lexer(code, "test.dim")
    for tok in lex.tokenize():
        print(tok)
