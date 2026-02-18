# Dim POC Lexer (Python)

import enum
import re

class TokenType(enum.Enum):
    IDENTIFIER = 1
    NUMBER = 2
    STRING = 3
    COLON = 4
    INDENT = 5
    DEDENT = 6
    NEWLINE = 7
    KEYWORD = 8
    OPERATOR = 9
    EOF = 10

class Lexer:
    def __init__(self, source):
        self.source = source
        self.pos = 0
        self.indent_stack = [0]
        self.tokens = []
        self.keywords = {"fn", "let", "mut", "struct", "prompt", "role", "match", "if", "else", "while", "for", "in", "await", "return"}

    def tokenize(self):
        lines = self.source.splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            # 1. Handle Indentation
            indent = len(line) - len(line.lstrip())
            if indent > self.indent_stack[-1]:
                self.indent_stack.append(indent)
                self.tokens.append((TokenType.INDENT, None))
            elif indent < self.indent_stack[-1]:
                while indent < self.indent_stack[-1]:
                    self.indent_stack.pop()
                    self.tokens.append((TokenType.DEDENT, None))
            
            # 2. Basic Tokenization (Regex approach for POC)
            content = line.strip()
            # Split by identifiers, numbers, strings, and operators
            parts = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|"(?:\\.|[^"\\])*"|==|!=|<=|>=|->|[=+\-*/<>():,\[\]]', content)
            for p in parts:
                if p in self.keywords:
                    self.tokens.append((TokenType.KEYWORD, p))
                elif p.startswith('"'):
                    self.tokens.append((TokenType.STRING, p[1:-1]))
                elif p.isdigit():
                    self.tokens.append((TokenType.NUMBER, int(p)))
                elif p == ":":
                    self.tokens.append((TokenType.COLON, None))
                elif p in {"+", "-", "*", "/", "<", ">", "==", "!=", "<=", ">=", "=", "(", ")", "[", "]", ",", "->"}:
                    self.tokens.append((TokenType.OPERATOR, p))
                else:
                    self.tokens.append((TokenType.IDENTIFIER, p))
                    
            self.tokens.append((TokenType.NEWLINE, None))
            
            
        # 3. Handle trailing dedents at EOF
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.tokens.append((TokenType.DEDENT, None))
            
        self.tokens.append((TokenType.EOF, None))
        return self.tokens

# Testing the POC Lexer
if __name__ == "__main__":
    code = """
fn hello():
    let x = 10
    prompt Greet:
        role user: "Hello"
"""
    lexer = Lexer(code)
    for t in lexer.tokenize():
        print(t)
