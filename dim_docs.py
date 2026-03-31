# dim_docs.py — Documentation Generator for Dim
#
# Generates documentation from Dim source files.

import os
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class DocComment:
    content: str
    line: int
    kind: str  # function, struct, enum, trait, etc.


@dataclass
class FunctionDoc:
    name: str
    params: List[str]
    return_type: str
    doc: str
    examples: List[str] = field(default_factory=list)


@dataclass
class StructDoc:
    name: str
    fields: List[str]
    doc: str
    methods: List[FunctionDoc] = field(default_factory=list)


@dataclass
class TraitDoc:
    name: str
    methods: List[str]
    doc: str


class DocParser:
    def __init__(self, source: str, filename: str):
        self.source = source
        self.filename = filename
        self.lines = source.split("\n")

    def parse_doc_comments(self) -> List[DocComment]:
        comments = []

        in_doc_comment = False
        doc_content = ""
        doc_line = 0

        for i, line in enumerate(self.lines):
            stripped = line.strip()

            if stripped.startswith("#"):
                if in_doc_comment:
                    doc_content += " " + stripped[1:].strip()
                else:
                    in_doc_comment = True
                    doc_content = stripped[1:].strip()
                    doc_line = i + 1
            else:
                if in_doc_comment and doc_content:
                    comments.append(DocComment(doc_content, doc_line, ""))
                in_doc_comment = False
                doc_content = ""

        return comments

    def parse_functions(self) -> List[FunctionDoc]:
        funcs = []

        fn_pattern = re.compile(
            r"^fn\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*(\w+))?", re.MULTILINE
        )

        for match in fn_pattern.finditer(self.source):
            name = match.group(1)
            params_str = match.group(2)
            return_type = match.group(3) or "unit"

            params = []
            if params_str.strip():
                for p in params_str.split(","):
                    p = p.strip()
                    if ":" in p:
                        p = p.split(":")[0].strip()
                    if p:
                        params.append(p)

            doc = self._get_doc_for_position(match.start())

            funcs.append(
                FunctionDoc(name=name, params=params, return_type=return_type, doc=doc)
            )

        return funcs

    def parse_structs(self) -> List[StructDoc]:
        structs = []

        struct_pattern = re.compile(r"^struct\s+(\w+):", re.MULTILINE)

        for match in struct_pattern.finditer(self.source):
            name = match.group(1)
            doc = self._get_doc_for_position(match.start())

            fields = []
            start_pos = match.end()
            end_pos = self.source.find("\n\n", start_pos)
            if end_pos == -1:
                end_pos = len(self.source)

            block = self.source[start_pos:end_pos]

            field_pattern = re.compile(r"^\s+(\w+):\s*(\w+)", re.MULTILINE)
            for field_match in field_pattern.finditer(block):
                fields.append(f"{field_match.group(1)}: {field_match.group(2)}")

            structs.append(StructDoc(name=name, fields=fields, doc=doc, methods=[]))

        return structs

    def parse_enums(self) -> List[Dict[str, Any]]:
        enums = []

        enum_pattern = re.compile(r"^enum\s+(\w+):", re.MULTILINE)

        for match in enum_pattern.finditer(self.source):
            name = match.group(1)
            doc = self._get_doc_for_position(match.start())

            variants = []
            start_pos = match.end()
            end_pos = self.source.find("\n\n", start_pos)
            if end_pos == -1:
                end_pos = len(self.source)

            block = self.source[start_pos:end_pos]

            for line in block.split("\n"):
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    if stripped.endswith(","):
                        stripped = stripped[:-1]
                    if stripped:
                        variants.append(stripped)

            enums.append({"name": name, "variants": variants, "doc": doc})

        return enums

    def parse_traits(self) -> List[TraitDoc]:
        traits = []

        trait_pattern = re.compile(r"^trait\s+(\w+)(?:\[.*\])?:", re.MULTILINE)

        for match in trait_pattern.finditer(self.source):
            name = match.group(1)
            doc = self._get_doc_for_position(match.start())

            methods = []
            start_pos = match.end()
            end_pos = self.source.find("\n\n", start_pos)
            if end_pos == -1:
                end_pos = len(self.source)

            block = self.source[start_pos:end_pos]

            fn_pattern = re.compile(r"fn\s+(\w+)")
            for fn_match in fn_pattern.finditer(block):
                methods.append(fn_match.group(1))

            traits.append(TraitDoc(name=name, methods=methods, doc=doc))

        return traits

    def _get_doc_for_position(self, pos: int) -> str:
        line_num = self.source[:pos].count("\n")

        look_back = 5
        start_line = max(0, line_num - look_back)

        doc_lines = []
        for i in range(start_line, line_num):
            line = self.lines[i].strip()
            if line.startswith("#"):
                doc_lines.append(line[1:].strip())
            else:
                break

        return " ".join(doc_lines)


class DocGenerator:
    def __init__(self):
        self.modules: Dict[str, Any] = {}

    def generate_markdown(self, source: str, filename: str) -> str:
        parser = DocParser(source, filename)

        funcs = parser.parse_functions()
        structs = parser.parse_structs()
        enums = parser.parse_enums()
        traits = parser.parse_traits()

        md = f"# {filename.replace('.dim', '').title()}\n\n"

        if funcs:
            md += "## Functions\n\n"
            for func in funcs:
                md += f"### `{func.name}`\n\n"
                if func.doc:
                    md += f"{func.doc}\n\n"
                md += f"**Signature:** `{func.name}({', '.join(func.params)})` -> `{func.return_type}`\n\n"

        if structs:
            md += "## Structs\n\n"
            for struct in structs:
                md += f"### `{struct.name}`\n\n"
                if struct.doc:
                    md += f"{struct.doc}\n\n"
                if struct.fields:
                    md += "**Fields:**\n"
                    for field in struct.fields:
                        md += f"- `{field}`\n"
                    md += "\n"

        if enums:
            md += "## Enums\n\n"
            for enum in enums:
                md += f"### `{enum['name']}`\n\n"
                if enum["doc"]:
                    md += f"{enum['doc']}\n\n"
                if enum["variants"]:
                    md += "**Variants:**\n"
                    for variant in enum["variants"]:
                        md += f"- `{variant}`\n"
                    md += "\n"

        if traits:
            md += "## Traits\n\n"
            for trait in traits:
                md += f"### `{trait.name}`\n\n"
                if trait.doc:
                    md += f"{trait.doc}\n\n"
                if trait.methods:
                    md += "**Methods:**\n"
                    for method in trait.methods:
                        md += f"- `{method}`\n"
                    md += "\n"

        return md

    def generate_html(self, source: str, filename: str) -> str:
        md = self.generate_markdown(source, filename)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{filename}</title>
    <style>
        body {{ font-family: system-ui; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
{self._markdown_to_html(md)}
</body>
</html>"""
        return html

    def _markdown_to_html(self, md: str) -> str:
        lines = md.split("\n")
        html_lines = []

        for line in lines:
            if line.startswith("# "):
                html_lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                html_lines.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("- "):
                html_lines.append(f"<li>{line[2:]}</li>")
            elif line.startswith("**") and "**" in line[2:]:
                html_lines.append(f"<p><strong>{line[2:-2]}</strong></p>")
            elif line.strip():
                html_lines.append(f"<p>{line}</p>")

        return "\n".join(html_lines)


def run_docs(args: List[str]):
    if not args:
        print("Usage: dim docs <file.dim> [output] [--html]")
        return

    filepath = args[0]
    output = args[1] if len(args) > 1 else None
    html_mode = "--html" in args

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        gen = DocGenerator()

        if html_mode:
            docs = gen.generate_html(source, filepath)
        else:
            docs = gen.generate_markdown(source, filepath)

        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(docs)
            print(f"Documentation written to {output}")
        else:
            print(docs)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import sys

    run_docs(sys.argv[1:] if len(sys.argv) > 1 else [])
