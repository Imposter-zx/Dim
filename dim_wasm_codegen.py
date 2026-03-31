# dim_wasm_codegen.py — WebAssembly Codegen for Dim
#
# Generates LLVM IR for WASM targets and provides .wasm output.

import subprocess
import os
import tempfile
from typing import Optional

from dim_mir import MIRModule
from dim_mir_to_llvm import LLVMGenerator, PLATFORM_TRIPLES, PLATFORM_DATALAYOUTS


def generate_wasm(
    module: MIRModule, output_path: str, optimization: int = 3
) -> Optional[str]:
    gen = LLVMGenerator(platform="wasm32")
    llvm_ir = gen.generate(module)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
        f.write(llvm_ir)
        ll_path = f.name

    try:
        result = subprocess.run(
            [
                "clang",
                "-O" + str(optimization),
                "-target",
                "wasm32",
                "-nostdlib",
                "-Wl,--no-entry",
                "-Wl,--export-all",
                "-o",
                output_path,
                ll_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print("WASM compilation failed:", result.stderr)
            return None
        return output_path
    except FileNotFoundError:
        print("clang not found. Install LLVM with WASM target:")
        print("  clang --version (must support --target=wasm32)")
        return None
    except subprocess.TimeoutExpired:
        print("WASM compilation timed out")
        return None
    finally:
        os.unlink(ll_path)


def compile_to_wasm(input_file: str, output_file: str = None) -> Optional[str]:
    from dim_lexer import Lexer
    from dim_parser import Parser
    from dim_semantic import SemanticAnalyzer
    from dim_mir_lowering import lower_program
    from dim_module_resolver import ModuleResolver

    if not output_file:
        output_file = input_file.replace(".dim", ".wasm")

    with open(input_file, "r") as f:
        source = f.read()

    tokens = Lexer(source, input_file).tokenize()
    parser = Parser(tokens, source, input_file)
    ast = parser.parse_program()
    if parser.diag.has_errors:
        parser.diag.flush()
        return None

    resolver = ModuleResolver(input_file)
    sem = SemanticAnalyzer(source, input_file, resolver)
    resolver.resolve_program(ast, source, input_file)
    if not sem.analyze(ast):
        sem.diag.flush()
        return None

    module = lower_program(ast)
    return generate_wasm(module, output_file)


def emit_wasm(module: MIRModule, output_path: str) -> Optional[str]:
    return generate_wasm(module, output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dim_wasm_codegen.py <input.dim> [output.wasm]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    result = compile_to_wasm(input_file, output_file)
    if result:
        print(f"Compiled to {result}")
    else:
        print("Compilation failed")
        sys.exit(1)
