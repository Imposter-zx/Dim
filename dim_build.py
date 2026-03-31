# dim_build.py — Build System for Dim
#
# Build management, project scaffolding, and build configuration.

import os
import json
import subprocess
from typing import Optional, List, Dict, Any
from pathlib import Path


BUILD_TARGETS = {
    "native": {"platform": "windows", "output": "target/debug"},
    "release": {"platform": "windows", "opt": 3, "output": "target/release"},
    "wasm": {"platform": "wasm32", "output": "target/wasm"},
    "wasm-release": {"platform": "wasm32", "opt": 3, "output": "target/wasm-release"},
}


class BuildConfig:
    def __init__(self, name: str = "dim-project"):
        self.name = name
        self.version = "0.1.0"
        self.main = "main.dim"
        self.target = "native"
        self.include_dirs: List[str] = []
        self.link_args: List[str] = []
        self.defines: Dict[str, str] = {}
        self.warnings: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "main": self.main,
            "target": self.target,
            "include_dirs": self.include_dirs,
            "link_args": self.link_args,
            "defines": self.defines,
            "warnings": self.warnings,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "BuildConfig":
        cfg = BuildConfig(data.get("name", "dim-project"))
        cfg.version = data.get("version", "0.1.0")
        cfg.main = data.get("main", "main.dim")
        cfg.target = data.get("target", "native")
        cfg.include_dirs = data.get("include_dirs", [])
        cfg.link_args = data.get("link_args", [])
        cfg.defines = data.get("defines", {})
        cfg.warnings = data.get("warnings", [])
        return cfg


class BuildSystem:
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.config: Optional[BuildConfig] = None
        self._load_config()

    def _load_config(self):
        config_path = os.path.join(self.project_dir, "build.dim.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = BuildConfig.from_dict(json.load(f))
        else:
            self.config = BuildConfig()

    def save_config(self):
        config_path = os.path.join(self.project_dir, "build.dim.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def init(self, name: str, main: str = "main.dim"):
        self.config = BuildConfig(name)
        self.config.main = main
        self.save_config()

        main_path = os.path.join(self.project_dir, main)
        if not os.path.exists(main_path):
            with open(main_path, "w") as f:
                f.write(f"""# {name} — A Dim program

fn main() -> unit:
    println("Hello from {name}!")
    return none
""")

        print(f"Initialized project: {name}")
        print(f"  Main file: {main}")
        print(f"  Build config: build.dim.json")

    def build(self, target: Optional[str] = None) -> bool:
        if not self.config:
            print("Error: No project configuration. Run 'dim build init' first.")
            return False

        target = target or self.config.target
        build_info = BUILD_TARGETS.get(target, BUILD_TARGETS["native"])

        print(f"Building {self.config.name} ({target})...")

        main_file = os.path.join(self.project_dir, self.config.main)
        if not os.path.exists(main_file):
            print(f"Error: Main file not found: {main_file}")
            return False

        output_dir = os.path.join(
            self.project_dir, build_info.get("output", "target/debug")
        )
        os.makedirs(output_dir, exist_ok=True)

        output_exe = os.path.join(output_dir, f"{self.config.name}.exe")

        return self._compile(main_file, output_exe, build_info)

    def _compile(self, input_file: str, output_file: str, opts: Dict[str, Any]) -> bool:
        from dim_lexer import Lexer
        from dim_parser import Parser
        from dim_semantic import SemanticAnalyzer
        from dim_module_resolver import ModuleResolver
        from dim_mir_lowering import lower_program
        from dim_mir_to_llvm import LLVMGenerator

        try:
            with open(input_file, "r") as f:
                source = f.read()

            print(f"  Parsing {input_file}...")
            tokens = Lexer(source, input_file).tokenize()
            parser = Parser(tokens, source, input_file)
            ast = parser.parse_program()

            if parser.diag.has_errors:
                print("  Parse errors:")
                parser.diag.flush()
                return False

            print("  Type checking...")
            resolver = ModuleResolver(input_file)
            sem = SemanticAnalyzer(source, input_file, resolver)
            resolver.resolve_program(ast, source, input_file)

            if not sem.analyze(ast):
                print("  Type errors:")
                sem.diag.flush()
                return False

            print("  Generating LLVM IR...")
            module = lower_program(ast)

            platform = opts.get("platform", "windows")
            gen = LLVMGenerator(platform=platform)
            llvm_ir = gen.generate(module)

            print(f"  Compiling to {output_file}...")

            opt_level = opts.get("opt", 0)

            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
                f.write(llvm_ir)
                ll_path = f.name

            runtime_path = os.path.join(
                os.path.dirname(__file__), "runtime", "dim_runtime.c"
            )

            result = subprocess.run(
                ["clang", f"-O{opt_level}", "-o", output_file, ll_path, runtime_path],
                capture_output=True,
                text=True,
                timeout=120,
            )

            os.unlink(ll_path)

            if result.returncode != 0:
                print("  Compilation failed:")
                print(result.stderr)
                return False

            print(f"  Built: {output_file}")
            return True

        except Exception as e:
            print(f"  Build error: {e}")
            return False

    def clean(self):
        import shutil

        target_dir = os.path.join(self.project_dir, "target")
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
            print("Cleaned build artifacts")

    def run(self, args: Optional[List[str]] = None) -> bool:
        target = self.config.target if self.config else "native"
        build_info = BUILD_TARGETS.get(target, BUILD_TARGETS["native"])
        output_dir = build_info.get("output", "target/debug")
        exe_name = f"{self.config.name}.exe" if self.config else "main.exe"
        output_exe = os.path.join(self.project_dir, output_dir, exe_name)

        if not os.path.exists(output_exe):
            print("Project not built. Running build first...")
            if not self.build():
                return False

        print(f"Running {output_exe}...")

        result = subprocess.run(
            [output_exe] + (args or []), capture_output=True, text=True
        )

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

        return result.returncode == 0


def run_build(args: List[str]):
    if not args:
        print("Usage: dim build <command> [options]")
        print("Commands:")
        print("  init <name>     Initialize a new project")
        print("  [target]       Build the project")
        print("  run             Build and run")
        print("  clean           Remove build artifacts")
        return

    project_path = os.getcwd()
    builder = BuildSystem(project_path)

    cmd = args[0]

    if cmd == "init":
        name = args[1] if len(args) > 1 else "my-project"
        main = args[2] if len(args) > 2 else "main.dim"
        builder.init(name, main)

    elif cmd == "run":
        builder.run(args[1:] if len(args) > 1 else None)

    elif cmd == "clean":
        builder.clean()

    elif cmd in BUILD_TARGETS:
        builder.build(cmd)

    else:
        builder.build()


if __name__ == "__main__":
    import sys

    run_build(sys.argv[1:] if len(sys.argv) > 1 else [])
