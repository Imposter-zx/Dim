import sys
import argparse
import json
from dataclasses import asdict
from dim_poc_lexer import Lexer
from dim_parser import Parser
from dim_semantic import SemanticAnalyzer
from dim_ast import Node

def node_to_dict(obj):
    if isinstance(obj, Node):
        result = {"type": obj.__class__.__name__}
        for key, value in asdict(obj).items():
            result[key] = node_to_dict(value)
        return result
    elif isinstance(obj, list):
        return [node_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: node_to_dict(v) for k, v in obj.items()}
    else:
        return obj

def run_tests():
    test_cases = [
        {
            "name": "Simple Arithmetic",
            "code": "let x = 10 + 20 * 3"
        },
        {
            "name": "Function with Params",
            "code": """
fn add(a: int, b: int) -> int:
    let sum = a + b
    return sum
"""
        },
        {
            "name": "AI Prompt",
            "code": """
prompt Greet:
    role user: "Hello AI"
    role assistant: "Hello Human"
"""
        }
    ]
    
    for case in test_cases:
        print(f"Running test: {case['name']}...")
        try:
            lexer = Lexer(case['code'])
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse_program()
            analyzer = SemanticAnalyzer()
            analyzer.analyze(ast)
            print("  - OK")
        except Exception as e:
            print(f"  - FAILED: {e}")

def main():
    parser = argparse.ArgumentParser(description="Dim Compiler Frontend CLI")
    parser.add_argument("file", nargs="?", help="Source file to compile")
    parser.add_argument("--tokens", action="store_true", help="Show tokens")
    parser.add_argument("--ast", action="store_true", help="Show AST")
    parser.add_argument("--analyze", action="store_true", help="Run semantic analysis")
    parser.add_argument("--test", action="store_true", help="Run internal tests")
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
        return

    if not args.file:
        parser.print_help()
        return

    with open(args.file, "r") as f:
        source = f.read()

    lexer = Lexer(source)
    tokens = lexer.tokenize()
    
    if args.tokens:
        for t in tokens:
            print(t)
            
    p = Parser(tokens)
    try:
        ast = p.parse_program()
        if args.ast:
            print(json.dumps(node_to_dict(ast), indent=2))
            
        if args.analyze:
            analyzer = SemanticAnalyzer()
            analyzer.analyze(ast)
            print("Semantic analysis complete.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
