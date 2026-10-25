"""AST-level semantic checks for the model_new.py wrapper.

The wrapper's compute must reach exactly one allowed sink: the
evaluator-loaded torch.ops.custom_op operator for Ascend C, or the
sample's own kernel launcher for pyasc. Sink accepts both shapes.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from .rules import TENSOR_COMPUTE_METHODS

# nn names that are structure, not compute: constructing them is always fine.
_STRUCTURAL_NN_NAMES = {
    "Module",
    "Parameter",
    "ParameterList",
    "ParameterDict",
    "ModuleList",
    "ModuleDict",
}

# torch.<name>() calls allowed in the wrapper: allocation, metadata, and data
# movement/layout glue. Anything else under torch.* is treated as compute.
_TORCH_ALLOWED_CALLS = {
    "empty",
    "zeros",
    "ones",
    "full",
    "empty_like",
    "zeros_like",
    "ones_like",
    "full_like",
    "empty_strided",
    "tensor",
    "as_tensor",
    "scalar_tensor",
    "arange",
    "rand",
    "randn",
    "randint",
    "rand_like",
    "randn_like",
    "Tensor",
    "device",
    "no_grad",
    "inference_mode",
    "is_tensor",
    "is_floating_point",
    "numel",
    "manual_seed",
    "cat",
    "concat",
    "concatenate",
    "stack",
    "vstack",
    "hstack",
    "dstack",
    "split",
    "chunk",
    "unbind",
    "reshape",
    "transpose",
    "permute",
    "squeeze",
    "unsqueeze",
    "flatten",
    "unflatten",
    "clone",
    "detach",
    "narrow",
    "select",
    "expand",
    "repeat",
    "tile",
    "broadcast_to",
    "view_as",
    "movedim",
    "moveaxis",
    "swapaxes",
    "swapdims",
    "roll",
    "flip",
}

_TENSOR_COMPUTE_METHODS = set(TENSOR_COMPUTE_METHODS) | {
    "clamp_",
    "cross",
    "cumprod",
    "dot",
    "einsum",
    "hardsigmoid",
    "hardswish",
    "index_add",
    "logsumexp",
    "masked_fill",
    "masked_fill_",
    "masked_select",
    "max",
    "min",
    "mish",
    "pow_",
    "scatter_",
    "softplus",
    "softsign",
}

_BANNED_IMPORT_ROOTS = {"ctypes", "subprocess", "socket", "importlib"}

_DYNAMIC_CALLS = {
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
}

_SCALAR_BUILTINS = {
    "len",
    "int",
    "float",
    "str",
    "bool",
    "repr",
}
# These builtins are scalar only when every argument is scalar.
_SCALAR_IF_ARGUMENTS_ARE_SCALAR = {"round", "abs", "min", "max", "sum"}
_SCALAR_METHODS = {
    "numel",
    "nelement",
    "dim",
    "size",
    "item",
    "element_size",
    "bit_length",
    "__len__",
}
_ARITH_OPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)


@dataclass(frozen=True)
class Sink:
    """The one callable the wrapper is allowed to route compute through.

    attr_mode names the modules whose attributes may be the sink, so
    import kernel followed by kernel.run() is a sink call. direct names
    the callables that are already the sink, so from kernel import run
    followed by run() is one too. With no Sink the sink is
    torch.ops.custom_op, which keeps the Ascend C behaviour unchanged.
    """

    direct: frozenset[str] = frozenset()
    attr_mode: frozenset[str] = frozenset()
    kind: str = "custom_op"
    attrs: frozenset[str] = frozenset()

    def matches(self, path: str) -> bool:
        """Return True when a dotted path names the sink itself."""
        if path.startswith("self."):
            path = path[5:]
        if path in self.direct:
            return True
        if any(path.startswith(module + ".") for module in self.attr_mode):
            return True
        # Import aliases resolve a launcher to either its bare name or the
        # kernel.name path; both name the same callable.
        return "." in path and path.rsplit(".", 1)[-1] in self.direct

    def is_sink_module(self, name: str) -> bool:
        """Return True when name is a module that carries the sink."""
        return name in self.attr_mode

    def message(self) -> str:
        """Return the violation for a wrapper that never calls the sink."""
        if self.kind == "kernel":
            return (
                "never calls the kernel launcher — model_new.py must import "
                "kernel and call its launch function"
            )
        return (
            "never calls torch.ops.custom_op — the wrapper must call the "
            "evaluator-loaded Ascend C operator"
        )

    def unknown_message(self, path: str) -> str:
        """Return the violation for an unknown sink module attribute."""
        if self.kind == "kernel":
            return (
                f"kernel.{path.rsplit('.', 1)[-1]} is not a launcher that "
                "reaches an @asc.jit kernel in kernel.py — call the kernel "
                "launcher the sample defines"
            )
        return f"vendor op-plugin call: {path}()"

    def kernel_attr_message(self, path: str) -> str | None:
        """Return the violation for a kernel attribute that is not a launcher.

        Returns None unless this sink is a kernel module whose launcher
        set is known, so the check never guesses.
        """
        if self.kind != "kernel" or "." not in path:
            return None
        head, _, tail = path.partition(".")
        if head in self.attr_mode and tail not in self.attrs:
            return self.unknown_message(path)
        return None


################################ AST VISITOR #################################
class WrapperSemantics(ast.NodeVisitor):
    """AST checks for model_new.py."""

    def __init__(
        self,
        source: str,
        sink: Sink | None = None,
        require_sink: bool = True,
        scalar_arith: bool = True,
        module_scalars: set[str] | None = None,
        parameter_scalars: dict[str, set[str]] | None = None,
        scalar_functions: set[str] | None = None,
    ) -> None:
        """Parse source and populate violations.

        Args:
            source: model_new.py text.
            sink: Allowed compute sink; None means torch.ops.custom_op.
            require_sink: Whether the source must call its compute sink. Host
                helpers that only compute shapes may set this False.
            scalar_arith: Whether to apply the heuristic arithmetic and
                comparison rules. They need type inference, so host helpers
                checked in isolation disable them; the definitive rules
                (torch calls, dynamic execution, caching) always apply.
            module_scalars: Proven scalar constants visible to this scope.
            parameter_scalars: Parameters proven scalar by every caller.
            scalar_functions: Functions whose return values are proven scalar.
        """
        self.violations: list[str] = []
        self.aliases: dict[str, str] = {}
        self.sink = sink
        self.co_refs: set[str] = set()
        self.sink_refs: set[str] = set()
        self.holders: set[str] = set()
        self.scalar_refs: set[str] = set()
        self.tensor_refs: set[str] = set()
        self.calls_custom_op = False
        self.calls_sink = False
        self.scalar_arith = scalar_arith
        self.parameter_scalars = parameter_scalars or {}
        self.scalar_functions = set(scalar_functions or ())
        self.function_scalars: dict[str, set[str]] = {}
        self.scalar_returns: set[str] = set()
        tree = ast.parse(source)
        self._collect_aliases(tree)
        self.scalar_refs |= set(module_scalars or ())
        self._collect_bindings_to_fixpoint(tree)
        self._infer_helper_parameters(tree)
        self.visit(tree)
        if require_sink and not self._reached_sink():
            self.violations.append(self._sink_message())

    def _infer_helper_parameters(self, tree: ast.Module) -> None:
        """Infer helper parameters from every call in this module."""
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        helpers = {
            name
            for name in functions
            if name not in {"forward", "__call__", "__init__"}
        }
        original_violations = list(self.violations)
        changed = True
        while changed:
            changed = False
            self.function_scalars = {}
            self.scalar_returns = set()
            self.visit(tree)
            new_returns = self.scalar_returns - self.scalar_functions
            if new_returns:
                self.scalar_functions.update(new_returns)
                changed = True
            evidence: dict[str, list[set[str]]] = {name: [] for name in helpers}
            for caller, function in functions.items():
                known = self.function_scalars.get(caller, set())
                aliases: dict[str, str] = {}
                scope = ast.Module(body=function.body, type_ignores=[])
                for node in _scope_nodes(scope):
                    if isinstance(node, ast.Assign) and isinstance(
                        node.value, ast.Name
                    ):
                        target_name = aliases.get(node.value.id, node.value.id)
                        if target_name in helpers:
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    aliases[target.id] = target_name
                for call in _scope_nodes(scope):
                    if not isinstance(call, ast.Call):
                        continue
                    if isinstance(call.func, ast.Name):
                        name = aliases.get(call.func.id, call.func.id)
                    elif isinstance(call.func, ast.Attribute):
                        name = call.func.attr
                    else:
                        continue
                    if name not in helpers:
                        continue
                    saved_scalars, saved_tensors = self.scalar_refs, self.tensor_refs
                    self.scalar_refs, self.tensor_refs = known, set()
                    bound = bind_call_arguments(call, functions[name])
                    evidence[name].append(
                        {
                            param
                            for param, value in bound.items()
                            if value is not None and self._is_scalar(value)
                        }
                    )
                    self.scalar_refs, self.tensor_refs = saved_scalars, saved_tensors
            for name, sites in evidence.items():
                if not sites:
                    continue
                proven = set.intersection(*sites)
                old = self.parameter_scalars.get(name, set())
                if proven - old:
                    self.parameter_scalars[name] = old | proven
                    changed = True
        self.violations = original_violations

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check each function with its own parameter and local bindings."""
        fields = (
            "scalar_refs",
            "tensor_refs",
            "aliases",
            "co_refs",
            "sink_refs",
            "holders",
        )
        saved = {name: getattr(self, name).copy() for name in fields}
        scope = ast.Module(body=node.body, type_ignores=[])
        parameters = {
            arg.arg
            for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        }
        if node.args.vararg:
            parameters.add(node.args.vararg.arg)
        if node.args.kwarg:
            parameters.add(node.args.kwarg.arg)
        locals_ = parameters | {
            item.id
            for item in _scope_nodes(scope)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
        }
        for field in fields:
            value = getattr(self, field)
            if isinstance(value, dict):
                setattr(
                    self, field, {k: v for k, v in value.items() if k not in locals_}
                )
            else:
                setattr(self, field, value - locals_)
        trusted = self.parameter_scalars.get(node.name, set())
        self.scalar_refs |= trusted
        self.tensor_refs |= parameters - trusted - {"self", "cls"}
        self._collect_aliases(scope)
        self._collect_bindings_to_fixpoint(scope)
        self.function_scalars[node.name] = self.scalar_refs - self.tensor_refs
        returns = [item for item in _scope_nodes(scope) if isinstance(item, ast.Return)]
        if returns and all(
            item.value is not None and self._is_scalar(item.value) for item in returns
        ):
            self.scalar_returns.add(node.name)
        for statement in node.body:
            self.visit(statement)
        attributes = {
            field: {name for name in getattr(self, field) if name.startswith("self.")}
            for field in ("scalar_refs", "co_refs", "sink_refs", "holders")
        }
        for field, value in saved.items():
            setattr(self, field, value)
        for field, names in attributes.items():
            getattr(self, field).update(names)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _reached_sink(self) -> bool:
        """Return True when the wrapper called its allowed sink."""
        return self.calls_sink if self.sink is not None else self.calls_custom_op

    def _sink_message(self) -> str:
        """Return the missing-sink violation for the active sink."""
        return (
            self.sink.message()
            if self.sink is not None
            else (
                "never calls torch.ops.custom_op — the wrapper must call the "
                "evaluator-loaded Ascend C operator"
            )
        )

    def _resolve(self, node: ast.AST, unwrap_subscript: bool = False) -> str | None:
        """Return the dotted path for a name or attribute, if resolvable.

        Args:
            node: Expression to resolve.
            unwrap_subscript: Treat kernel_fn[i](...) as a call of
                kernel_fn, which is how a pyasc kernel is launched.
        """
        if unwrap_subscript:
            while isinstance(node, ast.Subscript):
                node = node.value
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = self._resolve(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def _ref_key(self, node: ast.AST) -> str | None:
        """Dotted key for assignment targets: name or attribute path."""
        if isinstance(node, ast.Name):
            return node.id
        return self._resolve(node) if isinstance(node, ast.Attribute) else None

    def _is_custom_op_path(self, path: str) -> bool:
        """Return True if path is the resolved sink, or an alias."""
        if self.sink is not None:
            if path in self.sink_refs:
                return True
            return self.sink.matches(path)
        if path == "torch.ops.custom_op" or path.startswith("torch.ops.custom_op."):
            return True
        return any(path == ref or path.startswith(ref + ".") for ref in self.co_refs)

    def _unknown_sink_attr(self, path: str) -> str | None:
        """Return the path when it is an unresolved kernel-module attribute."""
        if self.sink is None or "." not in path:
            return None
        base, _ = path.split(".", 1)
        if self.sink.is_sink_module(base) and path not in self.sink.direct:
            return path
        return None

    def _collect_aliases(self, tree: ast.AST) -> None:
        """Record import aliases and flag banned import roots."""
        for node in _scope_nodes(tree):
            if isinstance(node, ast.Import):
                self._record_import(node)
            elif isinstance(node, ast.ImportFrom):
                self._record_import_from(node)

    def _record_import(self, node: ast.Import) -> None:
        """Bind import names and reject custom_op / banned roots."""
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root == "custom_op":
                self._flag(
                    "import custom_op is not used; call "
                    "torch.ops.custom_op after the evaluator loads "
                    "libcustom_op.so"
                )
            if root in _BANNED_IMPORT_ROOTS:
                self._flag(f"banned import: {alias.name}")
            self.aliases[alias.asname or root] = alias.name if alias.asname else root

    def _record_import_from(self, node: ast.ImportFrom) -> None:
        """Bind from-imports and reject unauditable star imports."""
        module = node.module or ""
        if module.split(".")[0] == "custom_op":
            self._flag(
                "from custom_op import ... is not used; call "
                "torch.ops.custom_op after the evaluator loads "
                "libcustom_op.so"
            )
        if module.split(".")[0] in _BANNED_IMPORT_ROOTS:
            self._flag(f"banned import: from {module}")
        for alias in node.names:
            if alias.name == "*":
                if module.split(".")[0] in {"torch", "torch_npu"}:
                    self._flag(f"star import from {module} (unauditable)")
                continue
            self.aliases[alias.asname or alias.name] = (
                f"{module}.{alias.name}" if module else alias.name
            )

    def _collect_bindings(self, tree: ast.AST) -> None:
        """Bind scalars, custom-op aliases, and nn-layer holders."""
        for node in _scope_nodes(tree):
            parts = self._assignment_parts(node)
            if parts is None:
                continue
            targets, value = parts
            if len(targets) == 1 and isinstance(targets[0], (ast.Tuple, ast.List)):
                self._bind_unpacked(targets[0].elts, value)
                continue
            for target in targets:
                self._bind_one_target(target, value)

    def _collect_bindings_to_fixpoint(self, tree: ast.AST) -> None:
        """Repeat binding propagation until no new name is classified."""
        while True:
            state = (
                len(self.scalar_refs),
                len(self.co_refs),
                len(self.holders),
                len(self.tensor_refs),
            )
            self._collect_bindings(tree)
            if state == (
                len(self.scalar_refs),
                len(self.co_refs),
                len(self.holders),
                len(self.tensor_refs),
            ):
                tainted: set[str] = set()
                for node in _scope_nodes(tree):
                    parts = self._assignment_parts(node)
                    if parts is None:
                        continue
                    targets, value = parts
                    if self._is_scalar(value):
                        continue
                    for target in targets:
                        if isinstance(target, (ast.Tuple, ast.List)):
                            tainted.update(
                                item.id
                                for item in ast.walk(target)
                                if isinstance(item, ast.Name)
                                and isinstance(item.ctx, ast.Store)
                            )
                        elif isinstance(target, (ast.Name, ast.Attribute)):
                            key = self._ref_key(target)
                            if key is not None:
                                tainted.add(key)
                new_taint = tainted - self.tensor_refs
                if not new_taint:
                    return
                self.tensor_refs.update(new_taint)

    def _assignment_parts(self, node: ast.AST) -> tuple[list[ast.AST], ast.AST] | None:
        """Return assignment targets and value, or bind a range loop."""
        if isinstance(node, ast.Assign):
            return node.targets, node.value
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            return [node.target], node.value
        if isinstance(node, (ast.For, ast.comprehension)):
            self._bind_range_target(node)
        return None

    def _bind_range_target(self, node: ast.For | ast.comprehension) -> None:
        """Bind a loop target as scalar when its iterable is scalar-only.

        range(...) is the common case, but a loop over a tuple of constants or
        over a list of scalars is shape arithmetic too. A loop over anything
        unknown leaves the target unknown, so tensor elements cannot be read
        out into a scalar name this way.
        """
        if not self._is_scalar(node.iter):
            return
        for name_node in ast.walk(node.target):
            if isinstance(name_node, ast.Name):
                self.scalar_refs.add(name_node.id)

    def _bind_unpacked(self, elts: list[ast.AST], value: ast.AST) -> None:
        """Bind names unpacked from a scalar or a tuple of scalars."""
        if self._is_scalar(value):
            for elt in elts:
                if isinstance(elt, ast.Name):
                    self.scalar_refs.add(elt.id)
            return
        if not (
            isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(elts)
        ):
            return
        for tgt, val in zip(elts, value.elts, strict=True):
            if isinstance(tgt, ast.Name) and self._is_scalar(val):
                self.scalar_refs.add(tgt.id)

    def _bind_one_target(self, target: ast.AST, value: ast.AST) -> None:
        """Bind one assignment target as scalar, custom-op, or nn holder."""
        key = self._ref_key(target)
        if key is None:
            return
        if self._is_scalar(value):
            self.scalar_refs.add(key)
        if self._is_known_tensor(value):
            self.tensor_refs.add(key)
        path = self._resolve(value)
        if path and self._is_custom_op_path(path):
            self.co_refs.add(key)
            self.sink_refs.add(key)
        if self._is_nn_layer_call(value):
            self.holders.add(key)

    def _is_scalar(self, node: ast.AST) -> bool:
        """Return True for expressions that cannot carry tensor data."""
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Name):
            return node.id in self.scalar_refs and node.id not in self.tensor_refs
        if isinstance(node, ast.Attribute):
            return self._is_scalar_attribute(node)
        if isinstance(node, ast.Subscript):
            return self._is_scalar_subscript(node)
        if isinstance(node, ast.Call):
            return self._is_scalar_call(node)
        if isinstance(node, ast.BinOp):
            return (
                isinstance(node.op, _ARITH_OPS)
                and self._is_scalar(node.left)
                and self._is_scalar(node.right)
            )
        if isinstance(node, ast.UnaryOp):
            return isinstance(
                node.op, (ast.UAdd, ast.USub, ast.Invert)
            ) and self._is_scalar(node.operand)
        if isinstance(node, (ast.Tuple, ast.List)):
            return all(self._is_scalar(elt) for elt in node.elts)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return all(
                self._is_scalar(gen.iter) and self._is_scalar(gen.target)
                for gen in node.generators
            ) and self._is_scalar(node.elt)
        if isinstance(node, ast.DictComp):
            return (
                all(
                    self._is_scalar(gen.iter) and self._is_scalar(gen.target)
                    for gen in node.generators
                )
                and self._is_scalar(node.key)
                and self._is_scalar(node.value)
            )
        return False

    def _is_scalar_attribute(self, node: ast.Attribute) -> bool:
        """Return True for x.shape / x.sizes or a bound scalar attribute."""
        if node.attr in {"shape", "sizes"}:
            return True
        key = self._resolve(node)
        return key in self.scalar_refs if key else False

    def _is_scalar_subscript(self, node: ast.Subscript) -> bool:
        """Return True for x.shape[i], x.size(...)[...], or a scalar name."""
        base = node.value
        if isinstance(base, ast.Attribute) and base.attr in {"shape", "sizes"}:
            return True
        if isinstance(base, ast.Call):
            if isinstance(base.func, ast.Attribute) and base.func.attr == "size":
                return True
            if isinstance(base.func, ast.Name):
                return self._is_scalar_call(base)
        return isinstance(base, ast.Name) and base.id in self.scalar_refs

    def _is_scalar_call(self, node: ast.Call) -> bool:
        """Return True for builtins, tensor metadata methods, or math.*."""
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in _SCALAR_BUILTINS:
                return True
            if func.id in _SCALAR_IF_ARGUMENTS_ARE_SCALAR | {"range", "tuple", "list"}:
                return all(self._is_scalar(arg) for arg in node.args) and all(
                    kw.arg is not None and self._is_scalar(kw.value)
                    for kw in node.keywords
                )
            if func.id in self.scalar_functions:
                return True
            return False
        if isinstance(func, ast.Attribute):
            if func.attr in _SCALAR_METHODS:
                return True
            path = self._resolve(func)
            return bool(path and path.startswith("math."))
        return False

    def _is_known_tensor(self, node: ast.AST) -> bool:
        """Return True for values known to carry tensor data."""
        if isinstance(node, ast.Name):
            return node.id in self.tensor_refs or node.id in self.co_refs
        if isinstance(node, ast.Attribute):
            key = self._resolve(node)
            return bool(key and (key in self.tensor_refs or key in self.co_refs))
        if isinstance(node, ast.Call):
            path = self._resolve(node.func)
            return bool(path and self._is_custom_op_path(path))
        return False

    def _is_nn_layer_call(self, node: ast.AST) -> bool:
        """Return True for a non-structural nn.Layer(...) constructor."""
        if not isinstance(node, ast.Call):
            return False
        path = self._resolve(node.func)
        if not path or not path.startswith("torch.nn."):
            return False
        rest = path[len("torch.nn.") :]
        if rest.startswith(("functional", "init", "parameter")):
            return False
        return rest.split(".")[0] not in _STRUCTURAL_NN_NAMES

    def _is_holder_call(self, node: ast.Call) -> bool:
        """Return True when an nn layer is invoked as compute."""
        func = node.func
        if isinstance(func, (ast.Name, ast.Attribute)):
            key = self._ref_key(func)
            if key in self.holders:
                return True
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Subscript):
                    base_key = self._ref_key(func.value.value)
                    if base_key in self.holders:
                        return True
                if func.attr == "forward":
                    base_key = self._ref_key(func.value)
                    if base_key in self.holders:
                        return True
        return isinstance(func, ast.Call) and self._is_nn_layer_call(func)

    def _flag(self, message: str) -> None:
        """Record one human-readable violation."""
        self.violations.append(message)

    def visit_Call(self, node: ast.Call) -> None:
        """Allow the sink call; flag other compute or side effects."""
        func_path = self._resolve(node.func, unwrap_subscript=True)
        if func_path and self._is_custom_op_path(func_path):
            self.calls_custom_op = True
            self.calls_sink = True
            self.generic_visit(node)
            return
        unknown = self._unknown_sink_attr(func_path) if func_path else None
        if unknown is not None:
            self._flag(self.sink.unknown_message(unknown))
            self.generic_visit(node)
            return
        flagged = self._flag_dynamic_attribute_call(node)
        flagged = self._flag_holder_or_dynamic(node) or flagged
        if func_path is not None:
            flagged = self._flag_resolved_call(func_path) or flagged
        if (
            not flagged
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _TENSOR_COMPUTE_METHODS
        ):
            self._flag(
                f"tensor-method compute (.{node.func.attr}(...)) — "
                "compute must live in the device kernel"
            )
        self.generic_visit(node)

    def _flag_dynamic_attribute_call(self, node: ast.Call) -> bool:
        """Flag getattr-style dispatch to a tensor compute method."""
        func = node.func
        if not isinstance(func, ast.Call) or not func.args:
            return False
        if isinstance(func.func, ast.Name) and func.func.id == "getattr":
            if len(func.args) < 2:
                return False
            name = self._constant_string(func.args[1])
            if name in _TENSOR_COMPUTE_METHODS:
                self._flag(
                    f"getattr() dynamic dispatch to compute method .{name}() — "
                    "compute must live in the Ascend C kernel"
                )
                return True
            return False
        if isinstance(func.func, ast.Attribute) and func.func.attr in {
            "__getattribute__",
            "__getattr__",
        }:
            name = self._constant_string(func.args[0])
            if name in _TENSOR_COMPUTE_METHODS:
                self._flag(
                    f"dynamic dispatch to compute method .{name}() — "
                    "compute must live in the Ascend C kernel"
                )
                return True
        return False

    @staticmethod
    def _constant_string(node: ast.AST) -> str | None:
        """Return a literal string argument, or None."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _flag_holder_or_dynamic(self, node: ast.Call) -> bool:
        """Flag nn-layer invocation and dynamic/getattr torch access."""
        flagged = False
        if self._is_holder_call(node):
            self._flag(
                "calls an nn layer as a function — nn modules may only hold "
                "parameters; the compute must go through custom_op"
            )
            flagged = True
        if not isinstance(node.func, ast.Name):
            return flagged
        name = node.func.id
        if name in {"setattr", "delattr"} and len(node.args) >= 2:
            base = self._resolve(node.args[0])
            attribute = self._constant_string(node.args[1])
            if self._is_pyasc_torch_sync(f"{base}.{attribute}"):
                self._flag("Reassigns timing function (monkey patch detected)")
                return True
        if name in _DYNAMIC_CALLS:
            self._flag(f"dynamic code execution ({name}())")
            return True
        if name in _SCALAR_IF_ARGUMENTS_ARE_SCALAR and any(
            self._is_known_tensor(arg) for arg in node.args
        ):
            self._flag(
                f"{name}() on tensor arguments — tensor compute must "
                "live in the Ascend C kernel"
            )
            return True
        if name == "getattr" and node.args:
            base = self._resolve(node.args[0])
            if base and (base == "torch" or base.startswith(("torch.", "torch_npu"))):
                self._flag("getattr() on torch modules (dynamic op access)")
                return True
        return flagged

    def _is_pyasc_torch_sync(self, path: str | None) -> bool:
        """Identify the one Torch synchronization call allowed for pyasc."""
        return (
            self.sink is not None
            and self.sink.kind == "kernel"
            and path == "torch.npu.synchronize"
        )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Protect the pyasc synchronization boundary from reassignment."""
        if isinstance(node.ctx, (ast.Store, ast.Del)) and self._is_pyasc_torch_sync(
            self._resolve(node)
        ):
            self._flag("Reassigns timing function (monkey patch detected)")
        self.generic_visit(node)

    def _flag_resolved_call(self, func_path: str) -> bool:
        """Flag vendor, functional, and disallowed torch.* calls."""
        if self._is_pyasc_torch_sync(func_path):
            return False
        if func_path.startswith("torch_npu."):
            self._flag(f"vendor native op shortcut: {func_path}()")
            return True
        if func_path == "torch.ops.load_library" or (
            func_path.endswith(".load_library") and "torch.ops" in func_path
        ):
            self._flag(
                "torch.ops.load_library is reserved for the evaluator; "
                "ModelNew must only call the already-loaded custom_op"
            )
            return True
        if func_path.startswith("torch.ops."):
            self._flag(f"vendor op-plugin call: {func_path}()")
            return True
        if func_path.startswith("torch.nn.functional."):
            self._flag(f"torch.nn.functional compute: {func_path}()")
            return True
        if func_path.startswith("torch.nn."):
            return False
        if func_path == "torch" or func_path.startswith("torch."):
            parts = func_path.split(".")
            if len(parts) == 2:
                if parts[1] not in _TORCH_ALLOWED_CALLS:
                    self._flag(
                        f"torch.{parts[1]}() is not allowed in "
                        "model_new.py (only allocation and "
                        "data-movement glue are)"
                    )
                    return True
                return False
            self._flag(f"{func_path}() is not allowed in the sample")
            return True
        if func_path.startswith(("os.system", "os.popen", "os.exec", "os.spawn")):
            self._flag(f"host process execution: {func_path}()")
            return True
        return False

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Flag tensor arithmetic and @; allow integer shape math."""
        if not self.scalar_arith:
            self.generic_visit(node)
            return
        if isinstance(node.op, ast.MatMult):
            self._flag("@ (matmul) operator — compute must live in the device kernel")
        elif isinstance(node.op, _ARITH_OPS) and not (
            self._is_scalar(node.left) and self._is_scalar(node.right)
        ):
            self._flag(
                "arithmetic on non-scalar values — tensor compute "
                "must live in the device kernel (integer shape "
                "arithmetic is fine)"
            )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Flag in-place tensor arithmetic and @=."""
        if not self.scalar_arith:
            self.generic_visit(node)
            return
        if not isinstance(node.op, ast.MatMult):
            if not (self._is_scalar(node.target) and self._is_scalar(node.value)):
                self._flag(
                    "in-place arithmetic on non-scalar values — tensor compute "
                    "must live in the device kernel"
                )
        else:
            self._flag("@ (matmul) operator — compute must live in the device kernel")
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        """Flag unary arithmetic on non-scalar (tensor) values."""
        if not self.scalar_arith:
            self.generic_visit(node)
            return
        if isinstance(
            node.op, (ast.UAdd, ast.USub, ast.Invert)
        ) and not self._is_scalar(node.operand):
            self._flag(
                "unary arithmetic on a non-scalar value — tensor compute must "
                "live in the Ascend C kernel"
            )
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        """Flag tensor comparisons; allow scalar and is None checks."""
        if not self.scalar_arith:
            self.generic_visit(node)
            return
        if any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
            self.generic_visit(node)
            return
        operands = [node.left, *node.comparators]
        none_check = any(
            isinstance(item, ast.Constant) and item.value is None for item in operands
        )
        if not none_check and not all(self._is_scalar(o) for o in operands):
            self._flag(
                "comparison on non-scalar values — tensor comparisons are "
                "compute and belong in the device kernel"
            )
        self.generic_visit(node)


################################ AST VISITOR #################################


def bind_call_arguments(
    call: ast.Call, target: ast.FunctionDef | ast.AsyncFunctionDef
) -> dict[str, ast.expr | None]:
    """Bind explicit and default call arguments without trusting annotations."""
    positional = [*target.args.posonlyargs, *target.args.args]
    if (
        isinstance(call.func, ast.Attribute)
        and positional
        and positional[0].arg in {"self", "cls"}
    ):
        positional = positional[1:]
    parameters = [*positional, *target.args.kwonlyargs]
    bound: dict[str, ast.expr | None] = {arg.arg: None for arg in parameters}
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
        keyword.arg is None for keyword in call.keywords
    ):
        return bound
    defaults = target.args.defaults
    if defaults:
        bound.update(
            zip(
                [arg.arg for arg in positional[-len(defaults) :]], defaults, strict=True
            )
        )
    bound.update(
        zip(
            [arg.arg for arg in target.args.kwonlyargs],
            target.args.kw_defaults,
            strict=True,
        )
    )
    assigned: set[str] = set()
    for param, value in zip(positional, call.args):
        bound[param.arg] = value
        assigned.add(param.arg)
    for keyword in call.keywords:
        if keyword.arg not in bound:
            continue
        if keyword.arg in assigned:
            return {name: None for name in bound}
        bound[keyword.arg] = keyword.value
        assigned.add(keyword.arg)
    return bound


def _scope_nodes(tree: ast.AST):
    """Walk one lexical scope without entering nested definitions."""
    pending = list(ast.iter_child_nodes(tree))
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            pending.extend(ast.iter_child_nodes(node))


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
    # `_GOOD = [0]` is a mutable scalar slot used to remember which variant
    # worked; a container of tensors never qualifies, so this cannot launder
    # tensor math.
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_scalar_expr(element, known) for element in node.elts)
    # A dict of scalars is bookkeeping too (`_CALLS = {"n": 0}`); a dict holding
    # a tensor cannot qualify because its value fails the same test.
    if isinstance(node, ast.Dict):
        return all(
            element is not None and _is_scalar_expr(element, known)
            for element in node.values
        )
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


def _collect_module_scalars(
    tree: ast.Module, include_functions: bool = True
) -> set[str]:
    """Return module-level names a host function may treat as plain values.

    Constants and torch-free helpers are defined beside the launcher, but a
    launcher is checked in isolation, where they would look like unknown
    non-scalars and make legal host shape arithmetic look like tensor compute.
    Constant definitions may chain (CHUNK = CORES * BLOCK_LEN), so the pass
    repeats until it stops resolving new names. A helper or value that mentions
    torch is excluded, so it cannot launder tensor math.
    """
    scalars: set[str] = set()
    if include_functions:
        # A wrapper keeps only value constants: a torch-free helper name must
        # stay unknown so a call to it is still inspected.
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
