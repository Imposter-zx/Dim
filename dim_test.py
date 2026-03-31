# dim_test.py — Test Framework for Dim
#
# Built-in test framework similar to Rust's #[test]

import os
import subprocess
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TestCase:
    name: str
    file: str
    line: int
    is_ignored: bool = False


class TestRunner:
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.test_cases: List[TestCase] = []
        self.passed = 0
        self.failed = 0
        self.ignored = 0

    def discover_tests(self) -> int:
        self.test_cases = []
        
        for root, dirs, files in os.walk(self.project_dir):
            if "target" in root or ".git" in root:
                continue
            
            for fname in files:
                if not fname.endswith(".dim"):
                    continue
                
                fpath = os.path.join(root, fname)
                self._scan_file(fpath)
        
        return len(self.test_cases)

    def _scan_file(self, fpath: str):
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        
        rel_path = os.path.relpath(fpath, self.project_dir)
        
        test_fn_pattern = re.compile(r'^fn\s+(test_\w+)\s*\(', re.MULTILINE)
        
        for i, line in enumerate(content.split("\n"), 1):
            if test_fn_pattern.match(line):
                is_ignored = False
                prev_lines = "\n".join(content.split("\n")[max(0, i-5):i])
                if "@ignore" in prev_lines or "#[ignore]" in prev_lines:
                    is_ignored = True
                
                self.test_cases.append(TestCase(
                    name=test_fn_pattern.match(line).group(1),
                    file=rel_path,
                    line=i,
                    is_ignored=is_ignored
                ))

    def run_tests(self, filter_name: Optional[str] = None) -> bool:
        if not self.test_cases:
            print("No test cases found")
            return True
        
        print(f"running {len(self.test_cases)} tests\n")
        
        for tc in self.test_cases:
            if filter_name and filter_name not in tc.name:
                continue
            
            if tc.is_ignored:
                print(f"test {tc.name} ... ignored")
                self.ignored += 1
                continue
            
            print(f"test {tc.name} ... ", end="")
            
            fpath = os.path.join(self.project_dir, tc.file)
            result = self._run_test_file(fpath)
            
            if result:
                print("ok")
                self.passed += 1
            else:
                print("FAILED")
                self.failed += 1
        
        print(f"\ntest result: ok. {self.passed} passed; {self.failed} failed; {self.ignored} ignored")
        
        return self.failed == 0

    def _run_test_file(self, fpath: str) -> bool:
        try:
            from dim_lexer import Lexer
            from dim_parser import Parser
            from dim_semantic import SemanticAnalyzer
            from dim_module_resolver import ModuleResolver
            from dim_mir_lowering import lower_program
            from dim_mir_to_llvm import LLVMGenerator
            from dim_diagnostic import DiagnosticBag
            
            with open(fpath, "r", encoding="utf-8") as f:
                source = f.read()
            
            diag = DiagnosticBag(source, fpath)
            
            tokens = Lexer(source, fpath).tokenize()
            parser = Parser(tokens, source, fpath)
            ast = parser.parse_program()
            
            if parser.diag.has_errors:
                return False
            
            resolver = ModuleResolver(fpath)
            sem = SemanticAnalyzer(source, fpath, resolver)
            resolver.resolve_program(ast, source, fpath)
            
            if not sem.analyze(ast):
                return False
            
            module = lower_program(ast)
            gen = LLVMGenerator(platform="windows")
            llvm_ir = gen.generate(module)
            
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
                f.write(llvm_ir)
                ll_path = f.name
            
            runtime_path = os.path.join(os.path.dirname(__file__), "runtime", "dim_runtime.c")
            
            result = subprocess.run(
                ["clang", "-O0", "-o", tempfile.mktemp(suffix=".exe"), ll_path, runtime_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            os.unlink(ll_path)
            
            if result.returncode != 0:
                return False
            
            exe_path = tempfile.mktemp(suffix=".exe")
            result = subprocess.run([exe_path], capture_output=True, text=True, timeout=10)
            
            try:
                os.unlink(exe_path)
            except:
                pass
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"\n   error: {e}")
            return False

    fn run_all(self, filter_name: Optional[str] = None) -> bool:
        return self.run_tests(filter_name)


def run_tests(args: List[str]):
    project_path = os.getcwd()
    
    runner = TestRunner(project_path)
    
    count = runner.discover_tests()
    print(f"Discovered {count} test cases")
    
    filter_name = args[0] if args else None
    
    success = runner.run_all(filter_name)
    
    exit(0 if success else 1)


if __name__ == "__main__":
    import sys
    run_tests(sys.argv[1:] if len(sys.argv) > 1 else [])