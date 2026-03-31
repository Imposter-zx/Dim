# dim_bench.py — Benchmark Framework for Dim
#
# Performance benchmarking and profiling.

import time
import os
import subprocess
import statistics
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    total_time: float
    mean_time: float
    median_time: float
    min_time: float
    max_time: float
    std_dev: float
    ops_per_sec: float


@dataclass
class BenchmarkSuite:
    name: str
    benchmarks: List[Callable] = field(default_factory=list)


class Benchmark:
    def __init__(
        self, name: str, func: Callable, iterations: int = 100, warmup: int = 3
    ):
        self.name = name
        self.func = func
        self.iterations = iterations
        self.warmup = warmup

    def run(self) -> BenchmarkResult:
        for _ in range(self.warmup):
            self.func()

        times: List[float] = []

        for _ in range(self.iterations):
            start = time.perf_counter()
            self.func()
            end = time.perf_counter()
            times.append(end - start)

        return self._compute_results(times)

    def _compute_results(self, times: List[float]) -> BenchmarkResult:
        total = sum(times)
        mean = total / len(times)
        median = statistics.median(times)
        min_t = min(times)
        max_t = max(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        ops = 1.0 / mean if mean > 0 else 0.0

        return BenchmarkResult(
            name=self.name,
            iterations=self.iterations,
            total_time=total,
            mean_time=mean,
            median_time=median,
            min_time=min_t,
            max_time=max_t,
            std_dev=std,
            ops_per_sec=ops,
        )


class DimBenchmark:
    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def add(self, name: str, func: Callable, iterations: int = 100):
        bench = Benchmark(name, func, iterations)
        result = bench.run()
        self.results.append(result)

    def compare(self, other: "DimBenchmark") -> Dict[str, float]:
        comparison = {}

        my_results = {r.name: r for r in self.results}
        other_results = {r.name: r for r in other.results}

        for name in my_results:
            if name in other_results:
                my_time = my_results[name].mean_time
                other_time = other_results[name].mean_time
                if other_time > 0:
                    ratio = my_time / other_time
                    comparison[name] = ratio

        return comparison

    def print_results(self):
        print("\n" + "=" * 80)
        print("Benchmark Results")
        print("=" * 80)
        print(f"{'Name':<30} {'Mean':>12} {'Median':>12} {'StdDev':>12} {'Ops/s':>12}")
        print("-" * 80)

        for result in sorted(self.results, key=lambda r: r.mean_time):
            print(
                f"{result.name:<30} "
                f"{result.mean_time * 1000:>10.3f}ms "
                f"{result.median_time * 1000:>10.3f}ms "
                f"{result.std_dev * 1000:>10.3f}ms "
                f"{result.ops_per_sec:>10.0f}"
            )

        print("=" * 80)


def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def quick_sort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


def matrix_multiply(n: int = 100) -> int:
    a = [[i * n + j for j in range(n)] for i in range(n)]
    b = [[i * n + j for j in range(n)] for i in range(n)]
    c = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i][j] += a[i][k] * b[k][j]

    return c[0][0]


def string_concat(n: int = 10000) -> str:
    result = ""
    for i in range(n):
        result += str(i)
    return result


def run_builtin_benchmarks():
    bench = DimBenchmark()

    print("Running built-in benchmarks...")

    def fib_bench():
        fibonacci(20)

    bench.add("fibonacci(20)", fib_bench, iterations=100)

    def sort_bench():
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]
        quick_sort(arr)

    bench.add("quick_sort(15)", sort_bench, iterations=1000)

    def matrix_bench():
        matrix_multiply(50)

    bench.add("matrix_multiply(50)", matrix_bench, iterations=10)

    def str_bench():
        string_concat(1000)

    bench.add("string_concat(1000)", str_bench, iterations=100)

    bench.print_results()


def run_file_benchmark(filepath: str, iterations: int = 10):
    print(f"Running benchmark on {filepath}...")

    try:
        from dim_lexer import Lexer
        from dim_parser import Parser
        from dim_semantic import SemanticAnalyzer
        from dim_module_resolver import ModuleResolver
        from dim_mir_lowering import lower_program

        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        bench = DimBenchmark()

        def lex_bench():
            Lexer(source, filepath).tokenize()

        bench.add("lex", lex_bench, iterations=iterations)

        tokens = Lexer(source, filepath).tokenize()

        def parse_bench():
            Parser(tokens, source, filepath).parse_program()

        bench.add("parse", parse_bench, iterations=iterations)

        parser = Parser(tokens, source, filepath)
        ast = parser.parse_program()

        resolver = ModuleResolver(filepath)
        sem = SemanticAnalyzer(source, filepath, resolver)
        resolver.resolve_program(ast, source, filepath)

        def typecheck_bench():
            resolver = ModuleResolver(filepath)
            sem = SemanticAnalyzer(source, filepath, resolver)
            resolver.resolve_program(ast, source, filepath)
            sem.analyze(ast)

        bench.add("typecheck", typecheck_bench, iterations=iterations)

        def lower_bench():
            lower_program(ast)

        bench.add("mir_lower", lower_bench, iterations=iterations)

        bench.print_results()

    except Exception as e:
        print(f"Benchmark error: {e}")


def run_benchmarks(args: List[str]):
    if not args:
        run_builtin_benchmarks()
        return

    filepath = args[0]
    iterations = int(args[1]) if len(args) > 1 else 10

    run_file_benchmark(filepath, iterations)


if __name__ == "__main__":
    import sys

    run_benchmarks(sys.argv[1:] if len(sys.argv) > 1 else [])
