# dim_linter.py — Linter for Dim
#
# Static analysis and code quality checks for Dim programs.

import re
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum


class LintLevel(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


@dataclass
class LintIssue:
    level: LintLevel
    code: str
    message: str
    file: str
    line: int
    column: int


class Rule:
    def check(self, source: str, ast: Any) -> List[LintIssue]:
        return []


class NamingConventionRule(Rule):
    def check(self, source: str, ast: Any) -> List[LintIssue]:
        issues = []

        fn_pattern = re.compile(r"^fn\s+([a-z_][a-z0-9_]*)", re.MULTILINE)
        for match in fn_pattern.finditer(source):
            name = match.group(1)
            line = source[: match.start()].count("\n") + 1

            if not name.islower():
                issues.append(
                    LintIssue(
                        level=LintLevel.WARNING,
                        code="E001",
                        message=f"Function name '{name}' should be snake_case",
                        file="",
                        line=line,
                        column=match.start(),
                    )
                )

            if name.startswith("_"):
                issues.append(
                    LintIssue(
                        level=LintLevel.INFO,
                        code="I001",
                        message=f"Function '{name}' starts with underscore (private)",
                        file="",
                        line=line,
                        column=match.start(),
                    )
                )

        struct_pattern = re.compile(r"^struct\s+([A-Z][a-zA-Z0-9]*)", re.MULTILINE)
        for match in struct_pattern.finditer(source):
            name = match.group(1)
            line = source[: match.start()].count("\n") + 1

            if not name[0].isupper():
                issues.append(
                    LintIssue(
                        level=LintLevel.WARNING,
                        code="E002",
                        message=f"Struct name '{name}' should be PascalCase",
                        file="",
                        line=line,
                        column=match.start(),
                    )
                )

        return issues


class UnusedVariableRule(Rule):
    def check(self, source: str, ast: Any) -> List[LintIssue]:
        issues = []

        let_pattern = re.compile(r"let\s+(mut\s+)?([a-z_][a-z0-9_]*)", re.MULTILINE)
        varsDefined = set()

        for match in let_pattern.finditer(source):
            varsDefined.add(match.group(2))

        for_pattern = re.compile(r"for\s+([a-z_][a-z0-9_]*)\s+in", re.MULTILINE)
        for match in for_pattern.finditer(source):
            if match.group(1) in varsDefined:
                varsDefined.discard(match.group(1))

        assign_pattern = re.compile(r"([a-z_][a-z0-9_]*)\s*=", re.MULTILINE)
        for match in assign_pattern.finditer(source):
            var_name = match.group(1)
            if var_name in varsDefined:
                varsDefined.discard(var_name)

        for var in varsDefined:
            pattern = re.compile(rf"let\s+.*\b{var}\b", re.MULTILINE)
            match = pattern.search(source)
            if match:
                line = source[: match.start()].count("\n") + 1
                issues.append(
                    LintIssue(
                        level=LintLevel.INFO,
                        code="I002",
                        message=f"Variable '{var}' appears unused",
                        file="",
                        line=line,
                        column=match.start(),
                    )
                )

        return issues


class ComplexityRule(Rule):
    def check(self, source: str, ast: Any) -> List[LintIssue]:
        issues = []

        for match in re.finditer(r"\bfn\s+(\w+)", source):
            fn_name = match.group(1)
            start = match.start()
            line_num = source[:start].count("\n") + 1

            brace_count = 0
            max_nesting = 0
            current_nesting = 0

            remaining = source[start:]
            for char in remaining:
                if char == "{":
                    brace_count += 1
                    current_nesting += 1
                    max_nesting = max(max_nesting, current_nesting)
                elif char == "}":
                    brace_count -= 1
                    current_nesting = max(0, current_nesting - 1)
                    if brace_count == 0:
                        break

            if max_nesting > 5:
                issues.append(
                    LintIssue(
                        level=LintLevel.WARNING,
                        code="E003",
                        message=f"Function '{fn_name}' has complexity {max_nesting} (max 5)",
                        file="",
                        line=line_num,
                        column=0,
                    )
                )

        return issues


class TODOCommentRule(Rule):
    def check(self, source: str, ast: Any) -> List[LintIssue]:
        issues = []

        patterns = [
            (r"TODO", "TODO found"),
            (r"FIXME", "FIXME found"),
            (r"XXX", "XXX found"),
            (r"HACK", "HACK found"),
        ]

        for pattern, msg in patterns:
            for match in re.finditer(pattern, source, re.IGNORECASE):
                line = source[: match.start()].count("\n") + 1
                issues.append(
                    LintIssue(
                        level=LintLevel.INFO,
                        code="I003",
                        message=f"{msg} - should be addressed",
                        file="",
                        line=line,
                        column=match.start(),
                    )
                )

        return issues


class PrintDebugRule(Rule):
    def check(self, source: str, ast: Any) -> List[LintIssue]:
        issues = []

        debug_patterns = [
            (r"\bprint\s*\(", "print() found"),
            (r"\bprintln\s*\(", "println() found"),
        ]

        for pattern, msg in debug_patterns:
            for match in re.finditer(pattern, source):
                line = source[: match.start()].count("\n") + 1

                context_start = max(0, match.start() - 50)
                context = source[context_start : match.start()]

                if "fn main" not in context:
                    issues.append(
                        LintIssue(
                            level=LintLevel.HINT,
                            code="H001",
                            message=f"{msg} - consider removing for production",
                            file="",
                            line=line,
                            column=match.start(),
                        )
                    )

        return issues


class ImportOrderRule(Rule):
    def check(self, source: str, ast: Any) -> List[LintIssue]:
        issues = []

        import_lines = []
        for match in re.finditer(r"^import\s+", source, re.MULTILINE):
            line = source[: match.start()].count("\n")
            import_lines.append((line, match.start()))

        if len(import_lines) > 1:
            modules = []
            for line, pos in import_lines:
                module_match = re.search(r"import\s+([\w.]+)", source[pos : pos + 50])
                if module_match:
                    modules.append(module_match.group(1))

            sorted_modules = sorted(modules)
            if modules != sorted_modules:
                issues.append(
                    LintIssue(
                        level=LintLevel.INFO,
                        code="I004",
                        message="Imports not sorted alphabetically",
                        file="",
                        line=import_lines[0][0],
                        column=0,
                    )
                )

        return issues


class Linter:
    def __init__(self):
        self.rules: List[Rule] = [
            NamingConventionRule(),
            UnusedVariableRule(),
            ComplexityRule(),
            TODOCommentRule(),
            PrintDebugRule(),
            ImportOrderRule(),
        ]
        self.enabled_rules: Set[str] = set()

    def lint(self, source: str, filename: str = "") -> List[LintIssue]:
        all_issues = []

        for rule in self.rules:
            issues = rule.check(source, None)
            for issue in issues:
                issue.file = filename
            all_issues.extend(issues)

        all_issues.sort(key=lambda x: (x.line, x.column))

        return all_issues

    def lint_file(self, filepath: str) -> List[LintIssue]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
            return self.lint(source, filepath)
        except Exception as e:
            return [
                LintIssue(
                    level=LintLevel.ERROR,
                    code="E999",
                    message=f"Failed to read file: {e}",
                    file=filepath,
                    line=1,
                    column=0,
                )
            ]


def run_linter(args: List[str]):
    if not args:
        print("Usage: dim lint <file.dim>")
        return

    filepath = args[0]
    linter = Linter()
    issues = linter.lint_file(filepath)

    if not issues:
        print(f"✓ No issues found in {filepath}")
        return

    level_colors = {
        LintLevel.ERROR: "\033[31m",
        LintLevel.WARNING: "\033[33m",
        LintLevel.INFO: "\033[36m",
        LintLevel.HINT: "\033[32m",
    }
    reset = "\033[0m"

    errors = sum(1 for i in issues if i.level == LintLevel.ERROR)
    warnings = sum(1 for i in issues if i.level == LintLevel.WARNING)

    print(f"Linting {filepath}: {errors} errors, {warnings} warnings\n")

    for issue in issues:
        color = level_colors.get(issue.level, "")
        print(
            f"{color}{issue.level.value.upper()}[{issue.code}]{reset}: {issue.message}"
        )
        print(f"  --> {issue.file}:{issue.line}:{issue.column}")


if __name__ == "__main__":
    import sys

    run_linter(sys.argv[1:] if len(sys.argv) > 1 else [])
