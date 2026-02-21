# dim_token.py — Structured Token & Span for Dim Compiler
# Replaces raw (TokenType, value) tuples with proper dataclasses.

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Any


class TokenType(Enum):
    # Literals
    INTEGER    = auto()
    FLOAT      = auto()
    STRING     = auto()
    BOOL       = auto()

    # Identifiers & Keywords
    IDENTIFIER = auto()
    KEYWORD    = auto()

    # Punctuation
    COLON      = auto()
    COMMA      = auto()
    DOT        = auto()
    SEMICOLON  = auto()

    # Grouping
    LPAREN     = auto()
    RPAREN     = auto()
    LBRACKET   = auto()
    RBRACKET   = auto()
    LBRACE     = auto()
    RBRACE     = auto()

    # Operators
    PLUS       = auto()
    MINUS      = auto()
    STAR       = auto()
    SLASH      = auto()
    PERCENT    = auto()
    EQ         = auto()   # =
    EQEQ       = auto()   # ==
    NEQ        = auto()   # !=
    LT         = auto()   # <
    GT         = auto()   # >
    LTE        = auto()   # <=
    GTE        = auto()   # >=
    AND        = auto()   # and / &&
    OR         = auto()   # or  / ||
    NOT        = auto()   # not / !
    ARROW      = auto()   # ->
    FAT_ARROW  = auto()   # =>
    AMP        = auto()   # & (borrow)
    PIPE       = auto()   # |

    # Indentation (injected by lexer)
    INDENT     = auto()
    DEDENT     = auto()
    NEWLINE    = auto()

    # Meta
    EOF        = auto()
    UNKNOWN    = auto()


# Comprehensive keyword set for Dim
KEYWORDS: frozenset[str] = frozenset({
    "fn", "let", "mut", "const", "return",
    "if", "else", "elif", "while", "for", "in", "break", "continue",
    "match", "struct", "enum", "trait", "impl", "type",
    "prompt", "role", "output", "model", "deterministic",
    "async", "await", "spawn", "actor", "receive",
    "import", "from", "as", "pub", "priv",
    "unsafe", "verify",
    "and", "or", "not",
    "true", "false", "none",
    "self", "Self",
    "extends", "where",
})


@dataclass(frozen=True)
class Span:
    """Half-open byte range [start, end) with file context."""
    file: str
    line_start: int   # 1-indexed
    col_start: int    # 1-indexed
    line_end: int
    col_end: int

    def __repr__(self) -> str:
        if self.line_start == self.line_end:
            return f"{self.file}:{self.line_start}:{self.col_start}-{self.col_end}"
        return f"{self.file}:{self.line_start}:{self.col_start}-{self.line_end}:{self.col_end}"

    def merge(self, other: "Span") -> "Span":
        """Return the smallest span that covers both self and other."""
        return Span(
            file=self.file,
            line_start=min(self.line_start, other.line_start),
            col_start=min(self.col_start, other.col_start)
                if self.line_start == other.line_start
                else (self.col_start if self.line_start < other.line_start else other.col_start),
            line_end=max(self.line_end, other.line_end),
            col_end=max(self.col_end, other.col_end)
                if self.line_end == other.line_end
                else (self.col_end if self.line_end > other.line_end else other.col_end),
        )

    @staticmethod
    def dummy() -> "Span":
        return Span("<dummy>", 0, 0, 0, 0)


@dataclass
class Token:
    """A lexed token with type, value, and source location."""
    kind: TokenType
    value: Any          # str, int, float, bool, or None
    span: Span

    def __repr__(self) -> str:
        return f"Token({self.kind.name}, {self.value!r}, {self.span})"

    @property
    def is_keyword(self) -> bool:
        return self.kind == TokenType.KEYWORD

    def keyword_is(self, *words: str) -> bool:
        return self.kind == TokenType.KEYWORD and self.value in words
