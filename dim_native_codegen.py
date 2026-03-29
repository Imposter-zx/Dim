# dim_native_codegen.py — Native binary generation
import subprocess
import os
import tempfile
from typing import Optional


class NativeCodegen:
    def __init__(self, ll_file: str, output: str = "a.out"):
        self.ll_file = ll_file
        self.output = output

    def compile_to_object(self) -> str:
        obj_file = tempfile.mktemp(suffix=".o")
        result = subprocess.run(
            ["llc", "-filetype=obj", "-o", obj_file, self.ll_file],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"llc failed: {result.stderr}")
        return obj_file

    def link(self, obj_file: str) -> str:
        result = subprocess.run(
            ["gcc", "-o", self.output, obj_file],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"link failed: {result.stderr}")
        return self.output

    def compile(self) -> Optional[str]:
        try:
            obj_file = self.compile_to_object()
            return self.link(obj_file)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Install LLVM tools (llc, lld) to generate native binaries")
            return None
        except Exception as e:
            print(f"Native compilation failed: {e}")
            return None


def emit_native(llvm_ir: str, output: str = "a.out") -> Optional[str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
        f.write(llvm_ir)
        ll_file = f.name

    try:
        codegen = NativeCodegen(ll_file, output)
        return codegen.compile()
    finally:
        if os.path.exists(ll_file):
            os.unlink(ll_file)
