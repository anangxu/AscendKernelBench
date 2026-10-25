"""Static checks for the pyasc backend: kernel.py plus model_new.py.

One sample is two Python files. kernel.py holds the @asc.jit device
functions and a plain-Python host launcher; model_new.py imports
kernel and calls that launcher.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .python_ast import (
    _SCALAR_METHODS,
    _TENSOR_COMPUTE_METHODS,
    Sink,
    WrapperSemantics,
    _collect_module_scalars,
    _scope_nodes,
    bind_call_arguments,
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
    scalar_functions: set[str] = field(default_factory=set)
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
    """Return calls in one scope without entering nested definitions."""
    return [
        node
        for node in _scope_nodes(ast.Module(body=nodes, type_ignores=[]))
        if isinstance(node, ast.Call)
    ]


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
    head, tail = path.split(".", 1)
    if head not in module_names:
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
    facts.module_scalars = _collect_module_scalars(tree, include_functions=False)
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


def _module_scope_statements(tree: ast.Module) -> list[ast.stmt]:
    """Return the statements that run at module scope.

    A definition or a collection may sit inside a module-level conditional or
    try block, which is still module scope, so those blocks are walked. A
    function or class body is not module scope and is never entered.
    """
    found: list[ast.stmt] = []

    def visit(statements: list[ast.stmt]) -> None:
        for node in statements:
            found.append(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ExceptHandler):
                    visit(child.body)
                elif isinstance(child, ast.stmt):
                    visit([child])

    visit(list(tree.body))
    return found


def _function_collections(tree: ast.Module, names: set[str]) -> dict[str, set[str]]:
    """Return module-level names holding only functions named in names."""
    collections: dict[str, set[str]] = {}
    changed = True
    while changed:
        changed = False
        for node in _module_scope_statements(tree):
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
        for node in _module_scope_statements(tree):
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


def _call_sites(
    node: ast.FunctionDef, facts: KernelFacts
) -> list[tuple[set[str], ast.Call]]:
    """Return possible host callees and their complete call arguments."""
    names = set(facts.host_functions)
    aliases = {alias: {target} for alias, target in facts.function_aliases.items()}
    changed = True
    while changed:
        changed = False
        for statement in _scope_nodes(ast.Module(body=node.body, type_ignores=[])):
            if not isinstance(statement, ast.Assign):
                continue
            found = _names_of_kernels(
                statement.value, names, aliases, facts.function_collections
            )
            for target in statement.targets:
                if isinstance(target, ast.Name) and found - aliases.get(
                    target.id, set()
                ):
                    aliases.setdefault(target.id, set()).update(found)
                    changed = True
    sites: list[tuple[set[str], ast.Call]] = []
    for call in _iter_calls(node.body):
        target: ast.AST = call.func
        while isinstance(target, ast.Subscript):
            target = target.value
        if isinstance(target, ast.Name):
            head = target.id
        elif isinstance(target, ast.Attribute):
            head = target.attr
        elif (
            isinstance(target, ast.Call)
            and isinstance(target.func, ast.Name)
            and target.func.id == "getattr"
            and len(target.args) >= 2
            and isinstance(target.args[1], ast.Constant)
            and isinstance(target.args[1].value, str)
        ):
            head = target.args[1].value
        else:
            continue
        expanded = ({head} if head in names else set()) | aliases.get(head, set())
        expanded |= facts.function_collections.get(head, set())
        if expanded:
            sites.append((expanded & names, call))
    return sites


def _is_scalar_in(
    node: ast.AST | None, known: set[str], scalar_functions: set[str] | None = None
) -> bool:
    """Return True when an expression can only carry a plain number.

    Conservative by design: anything not recognised counts as unknown, so the
    helper receiving it is treated as receiving a tensor.
    """
    if node is None:
        return False
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id in known
    if isinstance(node, ast.Attribute):
        return node.attr in {"shape", "sizes"}
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in {"int", "float", "bool", "len"}:
                return True
            if func.id in (scalar_functions or set()):
                return True
            if func.id in {
                "max",
                "min",
                "abs",
                "sum",
                "round",
                "range",
                "tuple",
                "list",
            }:
                return all(
                    _is_scalar_in(arg, known, scalar_functions) for arg in node.args
                ) and all(
                    kw.arg is not None
                    and _is_scalar_in(kw.value, known, scalar_functions)
                    for kw in node.keywords
                )
            return False
        if isinstance(func, ast.Attribute):
            return func.attr in _SCALAR_METHODS
        return False
    if isinstance(node, ast.Subscript):
        return _is_scalar_in(node.value, known, scalar_functions)
    if isinstance(node, ast.UnaryOp):
        return _is_scalar_in(node.operand, known, scalar_functions)
    if isinstance(node, ast.BinOp):
        return _is_scalar_in(node.left, known, scalar_functions) and _is_scalar_in(
            node.right, known, scalar_functions
        )
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(
            _is_scalar_in(element, known, scalar_functions) for element in node.elts
        )
    return False


def _wrapper_calls(
    source: str, facts: KernelFacts
) -> dict[str, list[tuple[set[str], ast.Call]]]:
    """Find launcher calls and scalar arguments in the paired wrapper."""
    tree = ast.parse(source)
    semantics = WrapperSemantics(source, require_sink=False, scalar_arith=False)
    imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "kernel":
            imports.update(
                {alias.asname or alias.name: alias.name for alias in node.names}
            )
    calls: dict[str, list[tuple[set[str], ast.Call]]] = {}
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    for function in functions:
        known = semantics.function_scalars.get(function.name, set())
        for call in _iter_calls(function.body):
            if isinstance(call.func, ast.Name):
                name = imports.get(call.func.id, call.func.id)
            elif isinstance(call.func, ast.Attribute):
                name = call.func.attr
            else:
                continue
            name = facts.function_aliases.get(name, name)
            if name in facts.launcher_entries:
                calls.setdefault(name, []).append((known, call))
    return calls


def _reachable_from_launchers(
    facts: KernelFacts, entries: set[str] | None = None
) -> set[str]:
    """Return launcher entries plus every host function they can call."""
    reachable = set(facts.launcher_entries) if entries is None else set(entries)
    changed = True
    while changed:
        changed = False
        for name in tuple(reachable):
            node = facts.host_functions.get(name)
            if node is None:
                continue
            for callees, _args in _call_sites(node, facts):
                for callee in callees:
                    if callee in facts.host_functions and callee not in reachable:
                        reachable.add(callee)
                        changed = True
    return reachable


def _function_semantics(
    node: ast.FunctionDef, facts: KernelFacts, param_scalars: set[str]
) -> WrapperSemantics:
    """Analyze one function using proven parameter and return facts."""
    return WrapperSemantics(
        ast.unparse(_module_of(node)),
        require_sink=False,
        module_scalars=facts.module_scalars,
        parameter_scalars={node.name: param_scalars},
        scalar_functions=facts.scalar_functions,
    )


def _scalar_params_by_function(
    facts: KernelFacts,
    reachable: set[str],
    wrapper_calls: dict[str, list[tuple[set[str], ast.Call]]] | None = None,
) -> dict[str, set[str]]:
    """Infer scalars only when every reachable call supplies a scalar."""
    wrapper_calls = wrapper_calls or {}
    scalars: dict[str, set[str]] = {name: set() for name in reachable}
    incoming: dict[str, list[tuple[str, ast.Call]]] = {name: [] for name in reachable}
    for caller in reachable:
        node = facts.host_functions.get(caller)
        if node is not None:
            for callees, call in _call_sites(node, facts):
                for callee in callees & reachable:
                    incoming[callee].append((caller, call))
    changed = True
    while changed:
        changed = False
        known: dict[str, set[str]] = {}
        for name in reachable:
            node = facts.host_functions.get(name)
            if node is None:
                continue
            semantics = _function_semantics(node, facts, scalars[name])
            known[name] = semantics.function_scalars.get(name, set())
            if name in semantics.scalar_returns and name not in facts.scalar_functions:
                facts.scalar_functions.add(name)
                changed = True
        for name, calls in incoming.items():
            all_calls = [(known[caller], call) for caller, call in calls]
            all_calls += wrapper_calls.get(name, [])
            if not all_calls:
                continue
            target = facts.host_functions[name]
            common: set[str] | None = None
            for caller_scalars, call in all_calls:
                bound = bind_call_arguments(call, target)
                supplied = {
                    param
                    for param, value in bound.items()
                    if _is_scalar_in(value, caller_scalars, facts.scalar_functions)
                }
                common = supplied if common is None else common & supplied
            proven = (common or set()) - scalars[name]
            if proven:
                scalars[name].update(proven)
                changed = True
    return scalars


def _module_of(node: ast.FunctionDef) -> ast.Module:
    """Return a module holding only node, so it can be walked in isolation."""
    module = ast.Module(body=[node], type_ignores=[])
    return ast.fix_missing_locations(module)


def check_kernel_launchers(
    facts: KernelFacts, wrapper_source: str | None = None
) -> list[str]:
    """Check every host function: allocate and launch, never compute.

    Host functions are the launchers plus any plain helper they call. A
    helper that touches torch is reported where it is written, so tensor
    math cannot be moved into a helper to dodge the launcher rule.
    """
    violations: list[str] = []
    jit = frozenset(facts.jit_functions)
    # A launcher may hand its work to helpers; those helpers are inspected too,
    # with the scalar parameters its call sites actually pass.
    try:
        wrapper_calls = _wrapper_calls(wrapper_source, facts) if wrapper_source else {}
    except SyntaxError:
        wrapper_calls = {}
    reachable = _reachable_from_launchers(facts, set(wrapper_calls) or None)
    scalar_params = _scalar_params_by_function(facts, reachable, wrapper_calls)
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
                # Every host function reachable from a launcher obeys the
                # shape arithmetic rule, in both directions: the launcher that
                # holds the subscript and the helpers it calls are all checked,
                # so tensor arithmetic cannot hide one call deeper.
                scalar_arith=name in reachable,
                module_scalars=facts.module_scalars,
                parameter_scalars={name: scalar_params.get(name, set())},
                scalar_functions=facts.scalar_functions,
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
    violations.extend(check_kernel_launchers(facts, wrapper_source))
    violations.extend(
        f"kernel.py: {message}" for message in check_kernel_host_rules(kernel_source)
    )
    violations.extend(check_pyasc_model_new(wrapper_source, facts))
    return dedupe(violations)
