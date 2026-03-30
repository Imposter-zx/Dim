# dim_formatter.py — Code Formatter for Dim
#
# Formats Dim source code with consistent indentation and style.

import re
from typing import List, Tuple


class DimFormatter:
    INDENT_SIZE = 4
    INDENT_CHAR = " "

    def __init__(self, indent_size: int = 4):
        self.indent_size = indent_size

    def format(self, source: str) -> str:
        lines = source.split("\n")
        result: List[str] = []
        indent_level = 0
        prev_indent = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            if not stripped:
                result.append("")
                continue

            if stripped.startswith("#"):
                result.append(" " * (indent_level * self.indent_size) + stripped)
                continue

            current_indent = len(line) - len(line.lstrip())

            if stripped.startswith("}"):
                indent_level = max(0, indent_level - 1)
                current_indent = indent_level * self.indent_size

            if stripped.endswith(":"):
                result.append(" " * current_indent + stripped)
                indent_level += 1
            elif stripped.endswith("{"):
                result.append(" " * current_indent + stripped)
            elif stripped.startswith("}"):
                result.append(" " * current_indent + stripped)
                if not self._is_end_block(stripped):
                    indent_level = max(0, indent_level - 1)
            elif stripped.startswith("fn ") or stripped.startswith("async fn "):
                result.append(" " * current_indent + stripped)
            elif stripped.startswith("struct ") or stripped.startswith("enum "):
                result.append(" " * current_indent + stripped)
            elif stripped.startswith("trait ") or stripped.startswith("impl "):
                result.append(" " * current_indent + stripped)
            elif stripped.startswith("if ") or stripped.startswith("elif "):
                result.append(" " * current_indent + stripped)
            elif stripped.startswith("else:"):
                result.append(" " * current_indent + stripped)
            elif stripped.startswith("while "):
                result.append(" " * current_indent + stripped)
            elif stripped.startswith("for "):
                result.append(" " * current_indent + stripped)
            elif stripped.startswith("match "):
                result.append(" " * current_indent + stripped)
            elif stripped.startswith("try:"):
                result.append(" " * current_indent + stripped)
            elif stripped.startswith("catch ") or stripped.startswith("finally:"):
                result.append(" " * current_indent + stripped)
            elif (
                stripped.startswith("return ")
                or stripped.startswith("let ")
                or stripped.startswith("break")
                or stripped.startswith("continue")
            ):
                result.append(" " * current_indent + stripped)
            elif stripped.startswith("}"):
                result.append(" " * current_indent + stripped)
            else:
                result.append(" " * current_indent + stripped)

        return "\n".join(result)

    def _is_end_block(self, line: str) -> bool:
        end_keywords = [
            "fn ",
            "struct ",
            "enum ",
            "trait ",
            "impl ",
            "if ",
            "elif ",
            "else:",
            "while ",
            "for ",
            "match ",
            "try:",
            "catch ",
            "finally:",
        ]
        return any(line.strip().startswith(kw) for kw in end_keywords)


def format_file(path: str, output_path: str = None, indent_size: int = 4):
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    formatter = DimFormatter(indent_size)
    formatted = formatter.format(source)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted)
    else:
        print(formatted)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dim_formatter.py <file> [output]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    format_file(input_file, output_file)
