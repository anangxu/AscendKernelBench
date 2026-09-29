"""Static checks for the pyasc backend: kernel.py plus model_new.py.

One sample is two Python files. kernel.py holds the @asc.jit device
functions and a plain-Python host launcher; model_new.py imports
kernel and calls that launcher.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .python_ast import (
    _TENSOR_COMPUTE_METHODS,
    Sink,
    WrapperSemantics,
)
from .text import dedupe

KERNEL_FILE = "kernel.py"
JIT_DECORATOR = "asc.jit"

# Modules whose calls are compute shortcuts in a device body.
COMPUTE_MODULES = ("torch", "torch_npu", "numpy", "np", "aten", "aclnn", "aclop")

# Kernel roots imported by name; a subscript on one of them is a launch.
KERNEL_IMPORT_ROOTS = ("kernel",)


@dataclass
class KernelFacts:
    """What kernel.py declares, and the launchers model_new.py may call."""

    jit_functions: dict[str, ast.FunctionDef] = field(default_factory=dict)
    launches: set[str] = field(default_factory=set)
    launchers: dict[str, ast.FunctionDef] = field(default_factory=dict)
    launcher_entries: dict[str, ast.FunctionDef] = field(default_factory=dict)
    host_functions: dict[str, ast.FunctionDef] = field(default_factory=dict)
    module_scalars: set[str] = field(default_factory=set)
    kernel_modules: set[str] = field(default_factory=set)
    imported_kernels: set[str] = field(default_factory=set)
    kernel_collections: dict[str, set[str]] = field(default_factory=dict)
    function_collections: dict[str, set[str]] = field(default_factory=dict)
    function_aliases: dict[str, str] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    def sink_names(self) -> set[str]:
        """Return every kernel-module attribute a wrapper may call."""
        return set(self.launcher_entries) | self.imported_kernels


def _resolve(node: ast.AST) -> str | None:
    """Return the dotted path of a name or attribute chain, or None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _resolve(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _iter_calls(nodes: list[ast.AST]) -> list[ast.Call]:
    """Return every call under nodes, not entering nested definitions."""
    calls: list[ast.Call] = []
    for node in ast.walk(ast.Module(body=list(nodes), type_ignores=[])):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call):
            calls.append(node)
    return calls


def _module_scope(tree: ast.Module) -> set[str]:
    """Return module-level names: definitions, imports, and assignments."""
    scope: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                scope.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    scope.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name_node in ast.walk(target):
                    if isinstance(name_node, ast.Name):
                        scope.add(name_node.id)
    return scope


def _decorator_paths(node: ast.FunctionDef) -> set[str]:
    """Return the dotted names of a function's decorators."""
    paths: set[str] = set()
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        path = _resolve(target)
        if path:
            paths.add(path)
    return paths


def _collect_kernel_imports(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Return kernel module names and names imported from a kernel module."""
    modules: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in KERNEL_IMPORT_ROOTS:
                    modules.add(alias.asname or root)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] not in KERNEL_IMPORT_ROOTS:
                continue
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
    return modules, names


def _is_jit_function(node: ast.FunctionDef) -> bool:
    """Return True for a function decorated with asc.jit."""
    return any(path.endswith(JIT_DECORATOR) for path in _decorator_paths(node))


def _launch_base(node: ast.AST, module_names: set[str]) -> tuple[str, str] | None:
    """Return (base, tail) for a subscripted target that names a kernel.

    A bare name gives (name, ""); kernel.name and kernel["name"] give
    ("kernel", "name"). Anything else returns None.
    """
    while isinstance(node, ast.Subscript):
        # _KERNELS[mode][cores](...) leaves a selector subscript under the
        # launch subscript, so resolve through the selectors to the root.
        node = node.value
    if isinstance(node, ast.Name):
        return node.id, ""
    path = _resolve(node)
    if path is None or "." not in path:
        return None
    head, _, tail = path.split(".", 1)
    if head not in module_names or "." in tail:
        return None
    return head, tail


def _launch_sites(
    body: list[ast.AST],
    kernel_names: set[str],
    imported: dict[str, str],
    module_names: set[str],
) -> tuple[set[str], list[str]]:
    """Return JIT kernels launched here plus names that cannot be resolved."""
    launched: set[str] = set()
    unknown: list[str] = []
    for call in _iter_calls(body):
        if not isinstance(call.func, ast.Subscript):
            continue
        base = _launch_base(call.func.value, module_names)
        if base is None:
            continue
        _, tail = base
        if not tail:
            launched.add(base[0])
        elif tail in kernel_names:
            launched.add(tail)
        elif tail in imported:
            launched.add(imported[tail])
        else:
            unknown.append(tail)
    return launched, unknown


def _is_module_launch(call: ast.Call, module_names: set[str]) -> bool:
    """Return True for kernel.<name>(...) or kernel['x'](...) call shapes."""
    if isinstance(call.func, ast.Subscript):
        return _resolve(call.func.value) in module_names
    path = _resolve(call.func)
    if path is None or "." not in path:
        return False
    return path.split(".", 1)[0] in module_names


def _device_violations(name: str, node: ast.FunctionDef, scope: set[str]) -> list[str]:
    """Flag torch/NumPy compute inside one @asc.jit device function.

    Args:
        name: Function name for the message.
        node: The decorated function definition.
        scope: Module-level names, so asc.add is not read as a method.
    """
    aliases = {root: root for root in COMPUTE_MODULES}
    for child in ast.walk(node):
        if isinstance(child, ast.Import):
            for alias in child.names:
                root = alias.name.split(".")[0]
                if root in COMPUTE_MODULES:
                    aliases[alias.asname or root] = root
        elif isinstance(child, ast.ImportFrom):
            root = (child.module or "").split(".")[0]
            if root in COMPUTE_MODULES:
                for alias in child.names:
                    aliases[alias.asname or alias.name] = root
    violations: list[str] = []
    for call in _iter_calls(list(node.body)):
        target = call.func
        if isinstance(target, ast.Attribute) and target.attr in _TENSOR_COMPUTE_METHODS:
            head = _resolve(target.value)
            root = head.split(".")[0] if head else ""
            if root not in aliases and root not in scope:
                violations.append(
                    f"{name}: tensor-method compute (.{target.attr}(...)) "
                    "inside an @asc.jit device function"
                )
                continue
        path = _resolve(target)
        if path is None:
            continue
        root = path.split(".")[0]
        if root in aliases:
            violations.append(
                f"{name}: torch/NumPy compute inside an @asc.jit device "
                f"function: {path}()"
            )
    return dedupe(violations)


def check_kernel_source(source: str) -> KernelFacts:
    """Check kernel.py and report what its host launchers may be called as.

    Args:
        source: Raw kernel.py text.

    Returns:
        Facts about the sample, with violations for a file that declares
        no device kernel, never launches, or computes in a device body.
    """
    facts = KernelFacts()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        facts.violations.append(f"kernel.py does not parse as Python: {exc}")
        return facts

    module_names, imported_names = _collect_kernel_imports(tree)
    facts.kernel_modules = module_names
    facts.imported_kernels = imported_names
    # A kernel or launcher may sit inside a module-level if/try block, and a
    # launcher may delegate the launch to a helper, so every definition in the
    # file counts: scanning only tree.body reported legal files as having no
    # kernel at all.
    module_level = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]
    facts.module_scalars = _collect_module_scalars(tree)
    for node in module_level:
        if _is_jit_function(node):
            facts.jit_functions[node.name] = node
    # Needs the kernel names, so it runs after they are collected. Both maps
    # are filled the same way; one is restricted to device kernels, the other
    # also covers host launchers, which a dispatcher may also collect.
    facts.kernel_collections = _function_collections(tree, set(facts.jit_functions))
    facts.function_collections = _function_collections(
        tree, set(facts.jit_functions) | {node.name for node in module_level}
    )
    # `run = softsign_launch` publishes a launcher under a second name, which
    # is the name model_new.py imports.
    facts.function_aliases = _function_aliases(
        tree, {node.name for node in module_level}
    )

    if not facts.jit_functions:
        facts.violations.append(
            "kernel.py defines no @asc.jit device function — the kernel must "
            "be a pyasc JIT function"
        )

    jit_names = set(facts.jit_functions)
    imported = {name: name for name in imported_names if name in jit_names}
    scope = _module_scope(tree)
    for node in module_level:
        if node.name in jit_names:
            facts.violations.extend(_device_violations(node.name, node, scope))
            continue
        launched, unknown = _launch_sites(node.body, jit_names, imported, module_names)
        # A launcher may pick its kernel at run time (chosen = a if cond else b)
        # and subscript the local name. That still launches, so resolve those
        # aliases back to the kernels they can hold before filtering.
        launched = _resolve_kernel_aliases(
            launched, node, jit_names, facts.kernel_collections
        )
        facts.host_functions[node.name] = node
        if launched:
            facts.launches.update(launched)
            facts.launchers[node.name] = node
        calls_module = any(
            _is_module_launch(call, module_names) for call in _iter_calls(node.body)
        )
        for tail in dedupe(unknown):
            facts.violations.append(
                f"kernel.py line {node.lineno}: {node.name} subscripts "
                f"{tail}, which is not an @asc.jit kernel defined in this "
                "file; only kernel_fn[core_num](...) launches a kernel"
            )
        if calls_module and not launched:
            facts.violations.append(
                f"kernel.py line {node.lineno}: {node.name} calls into the "
                "kernel module but never subscripts a kernel; a bare "
                "kernel_fn(...) call does not compile or launch"
            )

    if not facts.launches:
        facts.violations.append(
            "kernel.py has no subscripted launch of a local @asc.jit kernel — "
            "a bare kernel_fn(...) call does not compile or launch"
        )
    facts.launcher_entries = _reachable_launchers(facts)
    return facts


def _resolve_kernel_aliases(
    launched: set[str],
    node: ast.FunctionDef,
    jit_names: set[str],
    collections: dict[str, set[str]] | None = None,
) -> set[str]:
    """Map locally aliased names back to the @asc.jit kernels they may hold.

    `chosen = kernel_a if cond else kernel_b` followed by `chosen[cores](...)`
    launches one of two kernels; without this the sample looks like it never
    launches anything.
    """
    aliases = _kernel_aliases(node, jit_names, collections or {})

    resolved: set[str] = set()
    for name in launched:
        if name in jit_names:
            resolved.add(name)
        resolved |= aliases.get(name, set())
        # _KERNELS[mode][cores](...) subscripts the collection itself.
        resolved |= (collections or {}).get(name, set())
    return resolved & jit_names


def _kernel_aliases(
    node: ast.FunctionDef,
    jit_names: set[str],
    collections: dict[str, set[str]] | None = None,
) -> dict[str, set[str]]:
    """Return local names that can hold an @asc.jit kernel at call time.

    Covers `chosen = a if cond else b` and `for kernel in _KERNELS:`, both of
    which launch through a local name rather than the kernel name.
    """
    collections = collections or {}
    aliases: dict[str, set[str]] = {}

    def record(target: ast.AST, value: ast.AST) -> bool:
        if not isinstance(target, ast.Name):
            return False
        names = _names_of_kernels(value, jit_names, aliases, collections)
        if names and not names <= aliases.get(target.id, set()):
            aliases[target.id] = aliases.get(target.id, set()) | names
            return True
        return False

    changed = True
    while changed:
        changed = False
        for stmt in ast.walk(_module_of(node)):
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                changed |= record(stmt.targets[0], stmt.value)
            elif isinstance(stmt, ast.For):
                for target in ast.walk(stmt.target):
                    changed |= record(target, stmt.iter)
    return aliases


def _kernel_collections(tree: ast.Module, jit_names: set[str]) -> dict[str, set[str]]:
    """Return module-level names holding only @asc.jit kernels."""
    return _function_collections(tree, jit_names)


def _function_collections(tree: ast.Module, names: set[str]) -> dict[str, set[str]]:
    """Return module-level names holding only functions named in names."""
    collections: dict[str, set[str]] = {}
    changed = True
    while changed:
        changed = False
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            found = _names_of_kernels(node.value, names, {}, collections)
            if found and not found <= collections.get(target.id, set()):
                collections[target.id] = collections.get(target.id, set()) | found
                changed = True
    return collections


def _function_aliases(tree: ast.Module, names: set[str]) -> dict[str, str]:
    """Return module-level names bound straight to a function of this file."""
    aliases: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target, value = node.targets[0], node.value
            if not isinstance(target, ast.Name) or not isinstance(value, ast.Name):
                continue
            bound = aliases.get(value.id, value.id)
            if bound in names and aliases.get(target.id) != bound:
                aliases[target.id] = bound
                changed = True
    return aliases


def _names_of_kernels(
    value: ast.AST,
    jit_names: set[str],
    aliases: dict[str, set[str]],
    collections: dict[str, set[str]] | None = None,
) -> set[str]:
    """Return the kernels an expression names, directly or through aliases.

    Callers that pass host function names instead of @asc.jit names get the
    same treatment, which is how a launcher collection is recognised.
    """
    collections = collections or {}
    if isinstance(value, ast.Name):
        if value.id in jit_names:
            return {value.id}
        return set(aliases.get(value.id, set())) | set(collections.get(value.id, set()))
    if isinstance(value, ast.IfExp):
        return _names_of_kernels(
            value.body, jit_names, aliases, collections
        ) | _names_of_kernels(value.orelse, jit_names, aliases, collections)
    if isinstance(value, (ast.Tuple, ast.List)):
        found: set[str] = set()
        for element in value.elts:
            found |= _names_of_kernels(element, jit_names, aliases, collections)
        return found
    if isinstance(value, ast.Dict):
        found = set()
        for element in value.values:
            if element is not None:
                found |= _names_of_kernels(element, jit_names, aliases, collections)
        return found
    return set()


def _call_root(func: ast.AST) -> str | None:
    """Return the name a call target is rooted at, through launch subscripts."""
    while isinstance(func, ast.Subscript):
        func = func.value
    return func.id if isinstance(func, ast.Name) else None


def _reachable_launchers(facts: KernelFacts) -> dict[str, ast.FunctionDef]:
    """Return the host functions a wrapper may call to reach a launch.

    The prompt lets a launcher delegate the launch to a helper, so the entry
    point model_new.py calls is not always the function that subscripts the
    kernel. Grow the set until no further host function calls into it.
    """
    entries = dict(facts.launchers)
    for alias, target in facts.function_aliases.items():
        if target in entries and alias not in entries:
            entries[alias] = entries[target]
    changed = True
    while changed:
        changed = False
        for name, node in facts.host_functions.items():
            if name in entries:
                continue
            called: set[str] = set()
            for call in _iter_calls(node.body):
                root = _call_root(call.func)
                if root:
                    called.add(root)
            # A launcher that dispatches over a module-level tuple of launch
            # variants reaches every launcher in that collection.
            for root in tuple(called):
                called |= facts.function_collections.get(root, set())
            if called & set(entries):
                entries[name] = node
                changed = True
    return entries


def _module_of(node: ast.FunctionDef) -> ast.Module:
    """Return a module holding only node, so it can be walked in isolation."""
    module = ast.Module(body=[node], type_ignores=[])
    return ast.fix_missing_locations(module)


def check_kernel_launchers(facts: KernelFacts) -> list[str]:
    """Check every host function: allocate and launch, never compute.

    Host functions are the launchers plus any plain helper they call. A
    helper that touches torch is reported where it is written, so tensor
    math cannot be moved into a helper to dodge the launcher rule.
    """
    violations: list[str] = []
    jit = frozenset(facts.jit_functions)
    for name, node in facts.host_functions.items():
        # An aliased kernel (chosen = a if cond else b) is still the sink the
        # launcher routes compute through, and the message must be the pyasc
        # one rather than the Ascend C default.
        aliases = _kernel_aliases(node, set(jit), facts.kernel_collections)
        # Routing compute through a module-level collection of kernels, or
        # through another entry point that launches, still reaches the device.
        direct = jit | frozenset(aliases) | frozenset(facts.kernel_collections)
        direct |= frozenset(facts.launcher_entries)
        for collection, members in facts.function_collections.items():
            if members and members <= set(facts.launcher_entries):
                direct |= {collection}
        sink = Sink(
            direct=direct,
            attr_mode=frozenset(facts.kernel_modules),
            kind="kernel",
        )
        try:
            semantics = WrapperSemantics(
                ast.unparse(_module_of(node)),
                sink=sink,
                require_sink=name in facts.launchers,
                scalar_arith=name in facts.launchers,
                module_scalars=facts.module_scalars,
            )
        except SyntaxError as exc:
            violations.append(f"kernel.py {name} does not parse: {exc}")
            continue
        violations.extend(
            f"kernel.py {name}: {message}" for message in semantics.violations
        )
    return dedupe(violations)


def check_pyasc_model_new(source: str, facts: KernelFacts) -> list[str]:
    """Check model_new.py against the launchers kernel.py declares.

    Args:
        source: Raw model_new.py text.
        facts: Result of check_kernel_source for the paired kernel.py.

    Returns:
        Deduplicated human-readable violations; empty means pass.
    """
    from .python_source import check_model_new

    sink = Sink(
        direct=frozenset(facts.launcher_entries),
        attr_mode=frozenset(facts.kernel_modules),
        kind="kernel",
    )
    return check_model_new(source, sink=sink)


def check_kernel_host_rules(source: str) -> list[str]:
    """Apply the host-side regex catalog to kernel.py.

    The AST rules cover launchers and helpers; this adds the patterns that
    are cheaper to match textually and that the Ascend C backend already
    relies on: vendor native ops, stream primitives, threads, timing
    tampering, result caching, and CPU or NumPy fallbacks. The blunt
    try/except and pass rules stay out because a host helper may legitimately
    contain them.
    """
    from .python_source import (
        CPU_FALLBACK_PATTERNS,
        NPU_NATIVE_PATTERNS,
        RESULT_CACHE_PATTERNS,
        STREAM_PATTERNS,
        THREAD_PATTERNS,
        TIMING_EVENT_PATCH_PATTERNS,
        PatternRule,
        prepare_python_source,
        run_rules,
    )

    rules = (
        PatternRule(
            NPU_NATIVE_PATTERNS,
            "Uses vendor native op shortcut",
            include_match=True,
        ),
        PatternRule(
            STREAM_PATTERNS,
            "Uses stream primitives (potential timing manipulation)",
        ),
        PatternRule(
            THREAD_PATTERNS,
            "Uses threading/multiprocessing (potential timing manipulation)",
        ),
        PatternRule(
            TIMING_EVENT_PATCH_PATTERNS,
            "Reassigns timing function (monkey patch detected)",
        ),
        PatternRule(
            RESULT_CACHE_PATTERNS,
            "Caches results across calls (outputs must depend on current inputs)",
        ),
        PatternRule(CPU_FALLBACK_PATTERNS, "Contains CPU/NumPy fallback pattern"),
    )
    return run_rules(prepare_python_source(source), rules)


def check_pyasc_sources(kernel_source: str, wrapper_source: str) -> list[str]:
    """Return every pyasc violation for one sample's two files.

    Args:
        kernel_source: Raw kernel.py text.
        wrapper_source: Raw model_new.py text.

    Returns:
        Deduplicated human-readable violations; empty means pass.
    """
    facts = check_kernel_source(kernel_source)
    violations = list(facts.violations)
    violations.extend(check_kernel_launchers(facts))
    violations.extend(
        f"kernel.py: {message}" for message in check_kernel_host_rules(kernel_source)
    )
    violations.extend(check_pyasc_model_new(wrapper_source, facts))
    return dedupe(violations)


def _is_scalar_expr(node: ast.AST, known: set[str]) -> bool:
    """Return True for literals and arithmetic over already-known scalars."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id in known
    if isinstance(node, ast.UnaryOp):
        return _is_scalar_expr(node.operand, known)
    if isinstance(node, ast.BinOp):
        return _is_scalar_expr(node.left, known) and _is_scalar_expr(node.right, known)
    return False


def _touches_torch(node: ast.AST) -> bool:
    """Return True when a body mentions torch or NumPy."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and inner.id in {
            "torch",
            "torch_npu",
            "numpy",
            "np",
        }:
            return True
        if isinstance(inner, ast.Attribute):
            root: ast.AST = inner
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in {
                "torch",
                "torch_npu",
                "numpy",
                "np",
            }:
                return True
    return False


def _collect_module_scalars(tree: ast.Module) -> set[str]:
    """Return module-level names a host function may treat as plain values.

    Constants and torch-free helpers are defined beside the launcher, but a
    launcher is checked in isolation, where they would look like unknown
    non-scalars and make legal host shape arithmetic look like tensor compute.
    Constant definitions may chain (CHUNK = CORES * BLOCK_LEN), so the pass
    repeats until it stops resolving new names. A helper or value that mentions
    torch is excluded, so it cannot launder tensor math.
    """
    scalars: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not _touches_torch(node):
            scalars.add(node.name)
    changed = True
    while changed:
        changed = False
        for node in tree.body:
            value = (
                node.value
                if isinstance(node, ast.AnnAssign)
                else (node.value if isinstance(node, ast.Assign) else None)
            )
            if value is None or _touches_torch(value):
                continue
            targets = (
                [node.target] if isinstance(node, ast.AnnAssign) else list(node.targets)
            )
            # `H, W = 128, 256` binds one scalar per element, so both sides are
            # unpacked instead of the assignment being skipped for its target.
            if (
                len(targets) == 1
                and isinstance(targets[0], (ast.Tuple, ast.List))
                and isinstance(value, (ast.Tuple, ast.List))
                and len(targets[0].elts) == len(value.elts)
            ):
                if all(_is_scalar_expr(elt, scalars) for elt in value.elts):
                    for element in targets[0].elts:
                        if isinstance(element, ast.Name) and element.id not in scalars:
                            scalars.add(element.id)
                            changed = True
                continue
            if not _is_scalar_expr(value, scalars):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in scalars:
                    scalars.add(target.id)
                    changed = True
    return scalars
