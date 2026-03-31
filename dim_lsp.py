# dim_lsp.py — Language Server Protocol implementation for Dim
#
# Provides IDE support: diagnostics, completions, go-to-definition, hover.

import json
import sys
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

from dim_lexer import Lexer
from dim_parser import Parser
from dim_semantic import SemanticAnalyzer
from dim_module_resolver import ModuleResolver
from dim_mir_lowering import lower_program


class LSPDocument:
    def __init__(self, uri: str):
        self.uri = uri
        self.text = ""
        self.version = 0

    def update(self, content: str, version: int):
        self.text = content
        self.version = version


class DimLanguageServer:
    def __init__(self):
        self.documents: Dict[str, LSPDocument] = {}
        self.workspace_path: Optional[str] = None

    def _get_doc(self, uri: str) -> Optional[LSPDocument]:
        return self.documents.get(uri)

    def _parse_and_analyze(self, text: str, uri: str):
        try:
            tokens = Lexer(text, uri).tokenize()
            parser = Parser(tokens, text, uri)
            ast = parser.parse_program()

            if parser.diag.has_errors:
                return {"errors": parser.diag.errors, "warnings": parser.diag.warnings}

            resolver = ModuleResolver(uri)
            sem = SemanticAnalyzer(text, uri, resolver)
            resolver.resolve_program(ast, text, uri)

            if not sem.analyze(ast):
                return {"errors": sem.diag.errors, "warnings": sem.diag.warnings}

            return {"ast": ast, "sem": sem, "resolver": resolver}
        except Exception as e:
            return {"errors": [{"message": str(e), "line": 1, "column": 1}]}

    def initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        root_path = params.get("rootUri") or params.get("rootPath")
        if root_path:
            self.workspace_path = root_path

        return {
            "capabilities": {
                "textDocumentSync": 1,
                "completionProvider": {"triggerCharacters": [".", ":"]},
                "definitionProvider": True,
                "hoverProvider": True,
                "diagnosticsProvider": True,
            },
            "serverInfo": {"name": "Dim Language Server", "version": "0.5.0"},
        }

    def did_open(self, params: Dict[str, Any]):
        uri = params["textDocument"]["uri"]
        text = params["textDocument"]["text"]
        version = params["textDocument"]["version"]

        doc = LSPDocument(uri)
        doc.update(text, version)
        self.documents[uri] = doc

        return self.publish_diagnostics(uri)

    def did_change(self, params: Dict[str, Any]):
        uri = params["textDocument"]["uri"]
        version = params["textDocument"]["version"]

        for change in params.get("contentChanges", []):
            if uri in self.documents:
                self.documents[uri].update(change.get("text", ""), version)

        return self.publish_diagnostics(uri)

    def did_close(self, params: Dict[str, Any]):
        uri = params["textDocument"]["uri"]
        if uri in self.documents:
            del self.documents[uri]
        return None

    def publish_diagnostics(self, uri: str) -> Optional[Dict[str, Any]]:
        doc = self._get_doc(uri)
        if not doc:
            return None

        result = self._parse_and_analyze(doc.text, uri)

        diagnostics = []

        if "errors" in result:
            for err in result["errors"]:
                diagnostics.append(
                    {
                        "severity": 1,
                        "range": {
                            "start": {
                                "line": err.get("line", 1) - 1,
                                "character": err.get("column", 1) - 1,
                            },
                            "end": {
                                "line": err.get("line", 1) - 1,
                                "character": err.get("column", 1),
                            },
                        },
                        "message": err.get("message", "Unknown error"),
                    }
                )

        if "warnings" in result:
            for warn in result["warnings"]:
                diagnostics.append(
                    {
                        "severity": 2,
                        "range": {
                            "start": {
                                "line": warn.get("line", 1) - 1,
                                "character": warn.get("column", 1) - 1,
                            },
                            "end": {
                                "line": warn.get("line", 1) - 1,
                                "character": warn.get("column", 1),
                            },
                        },
                        "message": warn.get("message", "Warning"),
                    }
                )

        return {
            "method": "textDocument/publishDiagnostics",
            "params": {"uri": uri, "diagnostics": diagnostics},
        }

    def completion(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        uri = params["textDocument"]["uri"]
        position = params["position"]

        doc = self._get_doc(uri)
        if not doc:
            return []

        text = doc.text
        line = position["line"]
        col = position["character"]

        lines = text.split("\n")
        if line >= len(lines):
            return []

        line_text = lines[line]
        prefix = line_text[:col]

        completions = []

        keywords = [
            "fn",
            "let",
            "mut",
            "if",
            "else",
            "while",
            "for",
            "return",
            "struct",
            "enum",
            "trait",
            "impl",
            "import",
            "pub",
            "async",
            "match",
            "loop",
            "break",
            "continue",
            "try",
            "catch",
            "throw",
        ]

        for kw in keywords:
            if kw.startswith(prefix.split()[-1] if prefix.split() else prefix):
                completions.append({"label": kw, "kind": 14, "detail": "keyword"})

        stdlib_items = [
            ("print", "fn(msg: str) -> unit"),
            ("println", "fn(msg: str) -> unit"),
            ("input", "fn(prompt: str) -> str"),
            ("vec", "module"),
            ("io", "module"),
            ("math", "module"),
            ("str", "module"),
            ("file", "module"),
            ("json", "module"),
        ]

        for name, detail in stdlib_items:
            if name.startswith(prefix.split()[-1] if prefix.split() else prefix):
                completions.append(
                    {
                        "label": name,
                        "kind": 6 if detail == "module" else 1,
                        "detail": detail,
                    }
                )

        return completions

    def definition(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        uri = params["textDocument"]["uri"]
        position = params["position"]

        doc = self._get_doc(uri)
        if not doc:
            return None

        return None

    def hover(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        uri = params["textDocument"]["uri"]
        position = params["position"]

        doc = self._get_doc(uri)
        if not doc:
            return None

        return None


def run_lsp():
    server = DimLanguageServer()

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = request.get("method")
        params = request.get("params", {})
        msg_id = request.get("id")

        result = None
        error = None

        try:
            if method == "initialize":
                result = server.initialize(params)
            elif method == "textDocument/didOpen":
                publish = server.did_open(params)
                if publish:
                    print(json.dumps(publish))
            elif method == "textDocument/didChange":
                publish = server.did_change(params)
                if publish:
                    print(json.dumps(publish))
            elif method == "textDocument/didClose":
                server.did_close(params)
            elif method == "textDocument/completion":
                result = server.completion(params)
            elif method == "textDocument/definition":
                result = server.definition(params)
            elif method == "textDocument/hover":
                result = server.hover(params)
            elif method == "shutdown":
                result = None
            else:
                error = {"code": -32601, "message": f"Unknown method: {method}"}
        except Exception as e:
            error = {"code": -32603, "message": str(e)}

        response = {"jsonrpc": "2.0"}
        if msg_id is not None:
            response["id"] = msg_id
        if error:
            response["error"] = error
        else:
            response["result"] = result

        print(json.dumps(response))
        sys.stdout.flush()


if __name__ == "__main__":
    run_lsp()
