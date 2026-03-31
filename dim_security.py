# dim_security.py — Security Analysis for Dim
#
# Taint analysis, capability model, and contract verification.

from typing import Set, Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class SecurityLevel(Enum):
    UNTRUSTED = "untrusted"  # External input, tainted
    SANITIZED = "sanitized"  # Checked/cleaned
    TRUSTED = "trusted"  # Safe to use
    SENSITIVE = "sensitive"  # Secrets, keys, passwords


class Capability(Enum):
    NET_READ = "net_read"  # Read from network
    NET_WRITE = "net_write"  # Write to network
    FILE_READ = "file_read"  # Read files
    FILE_WRITE = "file_write"  # Write files
    ENV_READ = "env_read"  # Read environment
    PROCESS_SPAWN = "process_spawn"  # Spawn processes
    SYSTEM_CALL = "system_call"  # Make system calls
    MEMORY_ACCESS = "memory_access"  # Direct memory


@dataclass
class TaintSource:
    name: str
    level: SecurityLevel
    description: str
    sanitizers: List[str] = field(default_factory=list)


@dataclass
class SecurityAttribute:
    taint_level: SecurityLevel = SecurityLevel.UNTRUSTED
    required_caps: Set[Capability] = field(default_factory=set)
    sanitizers_applied: List[str] = field(default_factory=list)
    data_classification: Optional[str] = None


class TaintAnalyzer:
    def __init__(self):
        self.taint_sources: Dict[str, TaintSource] = {}
        self.sanitize_functions: Dict[str, List[str]] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.taint_sources = {
            "input": TaintSource("input", SecurityLevel.UNTRUSTED, "User input"),
            "read_file": TaintSource(
                "read_file", SecurityLevel.UNTRUSTED, "File contents"
            ),
            "http_request": TaintSource(
                "http_request", SecurityLevel.UNTRUSTED, "HTTP response"
            ),
            "env_var": TaintSource(
                "env_var", SecurityLevel.UNTRUSTED, "Environment variable"
            ),
            "command_arg": TaintSource(
                "command_arg", SecurityLevel.UNTRUSTED, "Command line argument"
            ),
        }

        self.sanitize_functions = {
            "escape_html": ["html"],
            "escape_sql": ["sql"],
            "escape_shell": ["shell"],
            "sanitize_path": ["path"],
            "validate_email": ["email"],
            "validate_url": ["url"],
            "strip_tags": ["html"],
            "base64_decode": ["base64"],
        }

    def is_tainted(self, attr: SecurityAttribute) -> bool:
        return attr.taint_level == SecurityLevel.UNTRUSTED

    def requires_capability(self, attr: SecurityAttribute, cap: Capability) -> bool:
        return cap in attr.required_caps

    def apply_sanitizer(
        self, attr: SecurityAttribute, sanitizer: str
    ) -> SecurityAttribute:
        if sanitizer in self.sanitize_functions.get(sanitizer, []):
            attr.sanitizers_applied.append(sanitizer)
            attr.taint_level = SecurityLevel.SANITIZED
        return attr

    def check_sink(self, attr: SecurityAttribute, sink_name: str) -> List[str]:
        violations = []

        if self.is_tainted(attr):
            violations.append(f"Unsanitized data flows to '{sink_name}'")

        return violations


class CapabilityChecker:
    def __init__(self):
        self.tool_permissions: Dict[str, Set[Capability]] = {}
        self.function_caps: Dict[str, Set[Capability]] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.tool_permissions = {
            "http_get": {Capability.NET_READ},
            "http_post": {Capability.NET_WRITE},
            "read_file": {Capability.FILE_READ},
            "write_file": {Capability.FILE_WRITE},
            "get_env": {Capability.ENV_READ},
            "spawn_process": {Capability.PROCESS_SPAWN},
        }

    def check_permission(self, func_name: str, required: Set[Capability]) -> bool:
        granted = self.function_caps.get(func_name, set())
        return required.issubset(granted)

    def add_permission(self, func_name: str, caps: Set[Capability]):
        if func_name not in self.function_caps:
            self.function_caps[func_name] = set()
        self.function_caps[func_name].update(caps)

    def verify_tool(self, tool_name: str, call_caps: Set[Capability]) -> bool:
        declared = self.tool_permissions.get(tool_name, set())
        return call_caps.issubset(declared)


class ContractVerifier:
    def __init__(self):
        self.preconditions: Dict[str, Any] = {}
        self.postconditions: Dict[str, Any] = {}
        self.invariants: Dict[str, Any] = {}

    def add_precondition(self, func: str, condition: str):
        self.preconditions[func] = condition

    def add_postcondition(self, func: str, condition: str):
        self.postconditions[func] = condition

    def add_invariant(self, class_name: str, condition: str):
        if class_name not in self.invariants:
            self.invariants[class_name] = []
        self.invariants[class_name].append(condition)

    def verify_contract(
        self, func: str, args: Dict[str, Any], result: Any
    ) -> List[str]:
        violations = []

        precond = self.preconditions.get(func)
        if precond and not self._evaluate_condition(precond, args):
            violations.append(f"Precondition violated: {precond}")

        postcond = self.postconditions.get(func)
        if postcond:
            ctx = {**args, "result": result}
            if not self._evaluate_condition(postcond, ctx):
                violations.append(f"Postcondition violated: {postcond}")

        return violations

    def _evaluate_condition(self, cond: str, ctx: Dict[str, Any]) -> bool:
        try:
            for key, val in ctx.items():
                cond = cond.replace(key, str(val))
            return eval(cond, {"__builtins__": {}}, {})
        except:
            return True


class SecurityPolicy:
    def __init__(self):
        self.taint_analyzer = TaintAnalyzer()
        self.capability_checker = CapabilityChecker()
        self.contract_verifier = ContractVerifier()
        self.deny_by_default = True

    def analyze_function(self, func_name: str, args: Dict[str, Any]) -> List[str]:
        errors = []

        for arg_name, attr in args.items():
            if isinstance(attr, SecurityAttribute):
                if self.taint_analyzer.is_tainted(attr):
                    for cap in attr.required_caps:
                        if not self.capability_checker.check_permission(
                            func_name, {cap}
                        ):
                            errors.append(
                                f"Missing capability {cap.value} for {arg_name}"
                            )

        return errors

    def check_data_flow(
        self, source: str, sink: str, attr: SecurityAttribute
    ) -> List[str]:
        violations = []

        violations.extend(self.taint_analyzer.check_sink(attr, sink))

        if attr.required_caps:
            for cap in attr.required_caps:
                if not self.capability_checker.check_permission(sink, {cap}):
                    violations.append(f"Sink '{sink}' lacks capability {cap.value}")

        return violations


def create_taint_attribute(
    level: SecurityLevel, caps: Optional[List[Capability]] = None
) -> SecurityAttribute:
    return SecurityAttribute(taint_level=level, required_caps=set(caps or []))


def create_tool_policy(tool_name: str, permissions: List[str]) -> Dict[str, Any]:
    cap_map = {
        "NetRead": Capability.NET_READ,
        "NetWrite": Capability.NET_WRITE,
        "FileRead": Capability.FILE_READ,
        "FileWrite": Capability.FILE_WRITE,
        "EnvRead": Capability.ENV_READ,
        "ProcessSpawn": Capability.PROCESS_SPAWN,
    }

    caps = {cap_map[p] for p in permissions if p in cap_map}

    return {
        "tool": tool_name,
        "capabilities": [c.value for c in caps],
        "security_level": SecurityLevel.SANITIZED.value,
    }


def run_security_demo():
    policy = SecurityPolicy()

    tainted_input = create_taint_attribute(SecurityLevel.UNTRUSTED)

    args = {"user_input": tainted_input}
    errors = policy.analyze_function("process_user_input", args)

    print("Security Analysis:")
    if errors:
        for err in errors:
            print(f"  ✗ {err}")
    else:
        print("  ✓ No security violations")

    tool_policy = create_tool_policy("http_get", ["NetRead"])
    print(f"\nTool Policy: {tool_policy}")


if __name__ == "__main__":
    run_security_demo()
