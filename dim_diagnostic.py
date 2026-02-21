# dim_diagnostic.py — Structured Diagnostic System for Dim Compiler
#
# Replaces bare print() error calls with a proper Diagnostic pipeline that
# carries severity, error codes, source spans, hints, and pretty-printing.

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional
from dim_token import Span


class Severity(Enum):
    ERROR   = auto()
    WARNING = auto()
    NOTE    = auto()
    HINT    = auto()


# ANSI colour codes (gracefully no-op on Windows without VT support)
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_WHITE  = "\033[97m"

_SEV_COLOR = {
    Severity.ERROR:   _RED,
    Severity.WARNING: _YELLOW,
    Severity.NOTE:    _CYAN,
    Severity.HINT:    _GREEN,
}

_SEV_LABEL = {
    Severity.ERROR:   "error",
    Severity.WARNING: "warning",
    Severity.NOTE:    "note",
    Severity.HINT:    "hint",
}


@dataclass
class Label:
    """An annotated source region inside a diagnostic."""
    span: Span
    message: str
    primary: bool = True   # True → ^^^ underlining; False → --- underlining


@dataclass
class Diagnostic:
    """
    A single compiler diagnostic message.

    Example output:
        error[E0001]: undefined variable `foo`
          --> src/main.dim:5:9
           5 |     let x = foo + 1
           |             ^^^ undefined variable `foo`
        hint: did you mean `for`?
    """
    severity: Severity
    code: str                        # e.g. "E0001"
    message: str                     # Main human-readable message
    labels: List[Label] = field(default_factory=list)
    notes: List[str]   = field(default_factory=list)
    hints: List[str]   = field(default_factory=list)

    # Optionally attach the full source so we can render snippets
    _source_lines: Optional[List[str]] = field(default=None, repr=False)

    def with_source(self, source: str) -> "Diagnostic":
        self._source_lines = source.splitlines()
        return self

    def render(self, color: bool = True) -> str:
        lines: List[str] = []
        sev_col = _SEV_COLOR[self.severity] if color else ""
        reset   = _RESET if color else ""
        bold    = _BOLD  if color else ""

        # --- Header line ---
        sev_label = _SEV_LABEL[self.severity]
        lines.append(
            f"{bold}{sev_col}{sev_label}[{self.code}]{reset}{bold}: {self.message}{reset}"
        )

        # --- Source snippets for each label ---
        for label in self.labels:
            sp = label.span
            if sp.line_start == 0:
                continue
            arrow_col = _CYAN if color else ""
            lines.append(f"  {arrow_col}-->{reset} {sp.file}:{sp.line_start}:{sp.col_start}")

            if self._source_lines and 0 < sp.line_start <= len(self._source_lines):
                src_line = self._source_lines[sp.line_start - 1]
                line_num = str(sp.line_start)
                gutter   = " " * len(line_num)
                lines.append(f"   {arrow_col}{gutter} |{reset}")
                lines.append(f"   {arrow_col}{line_num} |{reset} {src_line}")

                # Underline
                col_s = sp.col_start - 1  # zero-indexed
                col_e = sp.col_end   - 1
                length = max(1, col_e - col_s)
                under_char = "^" if label.primary else "-"
                under_col  = sev_col if label.primary else _CYAN if color else ""
                underline  = " " * col_s + under_char * length
                lines.append(
                    f"   {arrow_col}{gutter} |{reset} "
                    f"{under_col}{underline} {label.message}{reset}"
                )
                lines.append(f"   {arrow_col}{gutter} |{reset}")

        # --- Notes & Hints ---
        for note in self.notes:
            note_col = _CYAN if color else ""
            lines.append(f"   {note_col}= note:{reset} {note}")
        for hint in self.hints:
            hint_col = _GREEN if color else ""
            lines.append(f"   {hint_col}= hint:{reset} {hint}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render(color=False)


class DiagnosticBag:
    """
    Collects all diagnostics emitted during a compilation pass.
    Each pass receives a bag and writes to it; the driver decides
    when to flush and whether to abort.
    """

    def __init__(self, source: str = "", filename: str = "<unknown>"):
        self._diags: List[Diagnostic] = []
        self._source = source
        self._filename = filename

    # ── Emit helpers ─────────────────────────────────────────────────────────

    def error(self, code: str, message: str,
              span: Optional[Span] = None,
              hints: Optional[List[str]] = None) -> Diagnostic:
        d = Diagnostic(
            severity=Severity.ERROR,
            code=code,
            message=message,
            hints=hints or [],
        )
        if span:
            d.labels.append(Label(span, message, primary=True))
        if self._source:
            d.with_source(self._source)
        self._diags.append(d)
        return d

    def warning(self, code: str, message: str,
                span: Optional[Span] = None,
                hints: Optional[List[str]] = None) -> Diagnostic:
        d = Diagnostic(
            severity=Severity.WARNING,
            code=code,
            message=message,
            hints=hints or [],
        )
        if span:
            d.labels.append(Label(span, message, primary=True))
        if self._source:
            d.with_source(self._source)
        self._diags.append(d)
        return d

    def note(self, message: str, span: Optional[Span] = None) -> Diagnostic:
        d = Diagnostic(severity=Severity.NOTE, code="N000", message=message)
        if span:
            d.labels.append(Label(span, message, primary=False))
        self._diags.append(d)
        return d

    # ── Query ─────────────────────────────────────────────────────────────────

    @property
    def has_errors(self) -> bool:
        return any(d.severity == Severity.ERROR for d in self._diags)

    @property
    def all(self) -> List[Diagnostic]:
        return list(self._diags)

    def flush(self, color: bool = True) -> None:
        """Print all diagnostics to stderr and clear the bag."""
        import sys
        for d in self._diags:
            print(d.render(color=color), file=sys.stderr)
        self._diags.clear()

    def __len__(self) -> int:
        return len(self._diags)


# ── Error Code Registry ────────────────────────────────────────────────────────
# Centralised so tooling (LSP, docs) can enumerate all codes.

ERROR_CODES = {
    # Lexer
    "E0001": "Unknown character",
    "E0002": "Unterminated string literal",
    "E0003": "Invalid number literal",
    # Parser
    "E0010": "Unexpected token",
    "E0011": "Expected expression",
    "E0012": "Mismatched parentheses",
    "E0013": "Missing colon after block header",
    # Name resolution
    "E0020": "Undefined variable",
    "E0021": "Undefined function",
    "E0022": "Undefined type",
    "E0023": "Duplicate definition",
    # Type checking
    "E0030": "Type mismatch",
    "E0031": "Cannot infer type",
    "E0032": "Wrong number of arguments",
    "E0033": "Return type mismatch",
    # Borrow checker
    "E0040": "Use of moved value",
    "E0041": "Cannot borrow as mutable — already borrowed as immutable",
    "E0042": "Cannot borrow as immutable — already borrowed as mutable",
    "E0043": "Borrow outlives owner (dangling reference)",
    "E0044": "Assignment to immutable binding",
    # AI/Prompt
    "E0050": "Prompt called in pure (non-effectful) context",
    "E0051": "Prompt output type mismatch",
    "E0052": "Non-deterministic prompt in deterministic context",
    # Capabilities / Taint
    "E0060": "Tainted value used in sensitive sink without sanitisation",
    "E0061": "Missing capability for system operation",
}
