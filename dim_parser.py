# Dim POC Parser (Python)

from dim_poc_lexer import Lexer, TokenType
from dim_ast import *

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset=0):
        if self.pos + offset >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[self.pos + offset]

    def consume(self, expected_type=None):
        token = self.peek()
        if expected_type and token[0] != expected_type:
            raise Exception(f"Expected {expected_type}, got {token[0]} at pos {self.pos}")
        self.pos += 1
        return token

    def parse_program(self):
        stmts = []
        while self.peek()[0] != TokenType.EOF:
            if self.peek()[0] == TokenType.NEWLINE:
                self.consume()
            else:
                stmt = self.parse_statement()
                if stmt:
                    stmts.append(stmt)
        return Program(stmts)

    def parse_statement(self):
        token = self.peek()
        if token[0] == TokenType.KEYWORD:
            if token[1] == "fn":
                return self.parse_function()
            elif token[1] == "let":
                return self.parse_let()
            elif token[1] == "if":
                return self.parse_if()
            elif token[1] == "while":
                return self.parse_while()
            elif token[1] == "for":
                return self.parse_for()
            elif token[1] == "prompt":
                return self.parse_prompt()
            elif token[1] == "return":
                return self.parse_return()
        
        # Default to expression statement (not in AST yet, so we just consume for now or wrap in Expression)
        expr = self.parse_expression()
        if self.peek()[0] == TokenType.NEWLINE:
            self.consume()
        return expr # In POC, expressions can be statements

    def parse_block(self):
        self.consume(TokenType.COLON)
        while self.peek()[0] == TokenType.NEWLINE:
            self.consume()
        self.consume(TokenType.INDENT)
        body = []
        while self.peek()[0] != TokenType.DEDENT and self.peek()[0] != TokenType.EOF:
            if self.peek()[0] == TokenType.NEWLINE:
                self.consume()
            else:
                body.append(self.parse_statement())
        if self.peek()[0] == TokenType.DEDENT:
            self.consume(TokenType.DEDENT)
        return body

    def parse_let(self):
        self.consume(TokenType.KEYWORD) # 'let'
        is_mut = False
        if self.peek()[1] == "mut":
            self.consume(TokenType.KEYWORD)
            is_mut = True
        
        name = self.consume(TokenType.IDENTIFIER)[1]
        self.consume(TokenType.OPERATOR) # '='
        value = self.parse_expression()
        if self.peek()[0] == TokenType.NEWLINE:
            self.consume()
        return LetStmt(name, is_mut, None, value)

    def parse_function(self):
        self.consume(TokenType.KEYWORD) # 'fn'
        name = self.consume(TokenType.IDENTIFIER)[1]
        self.consume(TokenType.OPERATOR) # '('
        
        params = []
        if self.peek()[1] != ")":
            while True:
                param_name = self.consume(TokenType.IDENTIFIER)[1]
                self.consume(TokenType.COLON)
                param_type = self.consume(TokenType.IDENTIFIER)[1] # Simple type for now
                params.append((param_name, param_type))
                if self.peek()[1] == ",":
                    self.consume() # consume ','
                else:
                    break
        
        self.consume(TokenType.OPERATOR) # ')'
        
        return_type = None
        if self.peek()[1] == "->":
            self.consume() # consume '->'
            return_type = self.consume(TokenType.IDENTIFIER)[1]
            
        body = self.parse_block()
        return FunctionDef(name, params, return_type, body)

    def parse_if(self):
        self.consume(TokenType.KEYWORD) # 'if'
        condition = self.parse_expression()
        then_branch = self.parse_block()
        else_branch = None
        if self.peek()[1] == "else":
            self.consume(TokenType.KEYWORD)
            else_branch = self.parse_block()
        return IfStmt(condition, then_branch, else_branch)

    def parse_while(self):
        self.consume(TokenType.KEYWORD) # 'while'
        condition = self.parse_expression()
        body = self.parse_block()
        return WhileStmt(condition, body)

    def parse_for(self):
        self.consume(TokenType.KEYWORD) # 'for'
        iterator = self.consume(TokenType.IDENTIFIER)[1]
        self.consume(TokenType.KEYWORD) # 'in'
        iterable = self.parse_expression()
        body = self.parse_block()
        return ForStmt(iterator, iterable, body)

    def parse_prompt(self):
        self.consume(TokenType.KEYWORD) # 'prompt'
        name = self.consume(TokenType.IDENTIFIER)[1]
        self.consume(TokenType.COLON)
        self.consume(TokenType.NEWLINE)
        self.consume(TokenType.INDENT)
        roles = []
        while self.peek()[0] != TokenType.DEDENT:
            if self.peek()[0] == TokenType.NEWLINE:
                self.consume()
                continue
            self.consume(TokenType.KEYWORD) # 'role'
            role_name = self.consume(TokenType.IDENTIFIER)[1]
            self.consume(TokenType.COLON)
            content = self.consume(TokenType.STRING)[1]
            roles.append((role_name, content))
            if self.peek()[0] == TokenType.NEWLINE:
                self.consume()
        self.consume(TokenType.DEDENT)
        return PromptDef(name, None, roles, None)

    def parse_return(self):
        self.consume(TokenType.KEYWORD) # 'return'
        value = None
        if self.peek()[0] != TokenType.NEWLINE and self.peek()[0] != TokenType.EOF and self.peek()[0] != TokenType.DEDENT:
            value = self.parse_expression()
        if self.peek()[0] == TokenType.NEWLINE:
            self.consume()
        return ReturnStmt(value)

    def parse_expression(self):
        # Basic recursive descent for binary ops (Simplified Pratt for POC)
        return self.parse_binary_expr(0)

    def parse_binary_expr(self, min_precedence):
        left = self.parse_primary()
        
        precedences = {"+": 1, "-": 1, "*": 2, "/": 2, "==": 0, "<": 0, ">": 0}
        
        while True:
            op_token = self.peek()
            if op_token[0] != TokenType.OPERATOR or op_token[1] not in precedences:
                break
            
            op = op_token[1]
            prec = precedences[op]
            if prec < min_precedence:
                break
            
            self.consume()
            right = self.parse_binary_expr(prec + 1)
            left = BinaryOp(left, op, right)
            
        return left

    def parse_primary(self):
        token = self.peek()
        if token[0] == TokenType.NUMBER:
            self.consume()
            return Literal(token[1])
        if token[0] == TokenType.STRING:
            self.consume()
            return Literal(token[1])
        if token[0] == TokenType.IDENTIFIER:
            name = self.consume()[1]
            if self.peek()[1] == "(": # Call
                self.consume()
                args = []
                while self.peek()[1] != ")":
                    args.append(self.parse_expression())
                    if self.peek()[1] == ",": self.consume()
                self.consume()
                return Call(Identifier(name), args)
            return Identifier(name)
        if token[1] == "(":
            self.consume()
            expr = self.parse_expression()
            self.consume(TokenType.OPERATOR) # ')'
            return expr
        raise Exception(f"Unexpected token {token}")

# POC Execution
if __name__ == "__main__":
    from dim_poc_lexer import Lexer
    code = """
fn main():
    let x = 42 + 10 * 2
    if x > 50:
        print("Large")
    else:
        print("Small")
    
    prompt AI:
        role system: "You are a helper"
        role user: "Hello"
"""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse_program()
    print(ast)
