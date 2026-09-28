"""AST-level semantic checks for the model_new.py wrapper.

The wrapper's compute must reach exactly one allowed sink: the
evaluator-loaded torch.ops.custom_op operator for Ascend C, or the
sample's own kernel launcher for pyasc. Sink accepts both shapes.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

# nn names that are structure, not compute: constructing them is always fine.
_STRUCTURAL_NN_NAMES = {
    "Module",
    "Parameter",
    "ParameterList",
    "ParameterDict",
    "ModuleList",
    "ModuleDict",
    "init",
    "parameter",
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

_TENSOR_COMPUTE_METHODS = {
    "matmul",
    "mm",
    "bmm",
    "addmm",
    "baddbmm",
    "dot",
    "cross",
    "mv",
    "ger",
    "outer",
    "einsum",
    "relu",
    "relu_",
    "sigmoid",
    "sigmoid_",
    "tanh",
    "tanh_",
    "gelu",
    "silu",
    "softmax",
    "log_softmax",
    "leaky_relu",
    "elu",
    "selu",
    "hardsigmoid",
    "hardswish",
    "softplus",
    "softsign",
    "mish",
    "add",
    "add_",
    "sub",
    "sub_",
    "mul",
    "mul_",
    "div",
    "div_",
    "pow",
    "pow_",
    "sum",
    "mean",
    "max",
    "min",
    "amax",
    "amin",
    "argmax",
    "argmin",
    "prod",
    "norm",
    "var",
    "std",
    "cumsum",
    "cumprod",
    "topk",
    "sort",
    "logsumexp",
    "clamp",
    "clamp_",
    "conv1d",
    "conv2d",
    "conv3d",
    "gather",
    "scatter",
    "scatter_",
    "index_select",
    "index_add",
    "masked_fill",
    "masked_fill_",
    "masked_select",
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
    "round",
    "abs",
    "min",
    "max",
    "sum",
    "str",
    "bool",
    "repr",
}
_SCALAR_METHODS = {"numel", "nelement", "dim", "size", "item", "__len__"}
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
            module_scalars: Names defined beside the source at module level,
                such as constants and torch-free helpers. A launcher is also
                checked in isolation, where those names would otherwise look
                like unknown non-scalars.
        """
        self.violations: list[str] = []
        self.aliases: dict[str, str] = {}
        self.sink = sink
        self.co_refs: set[str] = set()
        self.sink_refs: set[str] = set()
        self.holders: set[str] = set()
        self.scalar_refs: set[str] = set()
        self.calls_custom_op = False
        self.calls_sink = False
        self.scalar_arith = scalar_arith
        tree = ast.parse(source)
        self._collect_aliases(tree)
        self.local_funcs = self._collect_scalar_helpers(tree)
        self.defined_names = self._collect_defined_names(tree)
        # Names defined beside the source and scalar-annotated parameters must
        # be known before bindings are resolved: a launcher checked in
        # isolation writes `cores = CORE_LIMIT` or `total: int`, and resolving
        # those afterwards made legal shape arithmetic look like tensor math.
        self.scalar_refs |= set(module_scalars or ())
        self._collect_annotated_scalars(tree)
        for _ in range(3):
            self._collect_bindings(tree)
        self.visit(tree)
        if require_sink and not self._reached_sink():
            self.violations.append(self._sink_message())

    def _collect_annotated_scalars(self, tree: ast.Module) -> None:
        """Add parameters annotated as int, float, or bool to scalar_refs.

        The pyasc contract carries sizes as plain scalars, so `total: int` is
        shape arithmetic. Unannotated parameters stay unknown: they may be
        device tensors, and treating them as scalars would let a launcher add
        tensors without being reported.
        """
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for arg in [*node.args.args, *node.args.kwonlyargs]:
                if self._is_scalar_annotation(arg.annotation):
                    self.scalar_refs.add(arg.arg)

    @staticmethod
    def _is_scalar_annotation(annotation: ast.AST | None) -> bool:
        """Return True for an int, float, or bool annotation."""
        return isinstance(annotation, ast.Name) and annotation.id in {
            "int",
            "float",
            "bool",
        }

    def _collect_scalar_helpers(self, tree: ast.Module) -> set[str]:
        """Return local functions that never touch torch.

        A host tiling helper is plain integer arithmetic; a call to it must
        not be read as tensor compute. A helper that mentions torch at all is
        excluded, so it cannot launder tensor math past the arithmetic rule.
        Helper bodies are still visited independently, so their own torch use
        is reported where it happens.
        """
        helpers: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            touches_torch = False
            for inner in ast.walk(node):
                if isinstance(inner, ast.Name) and inner.id in {
                    "torch",
                    "torch_npu",
                    "numpy",
                    "np",
                }:
                    touches_torch = True
                    break
                if isinstance(inner, ast.Attribute):
                    path = self._resolve(inner)
                    if path and path.split(".")[0] in {
                        "torch",
                        "torch_npu",
                        "numpy",
                        "np",
                    }:
                        touches_torch = True
                        break
            if not touches_torch:
                helpers.add(node.name)
        return helpers

    def _collect_defined_names(self, tree: ast.Module) -> set[str]:
        """Return every name the module itself defines or imports.

        A launcher is also checked in isolation, where its module-level
        helpers are undefined here; such a name can only be an external
        integer helper, never tensor data the module produced.
        """
        names: set[str] = set(self.aliases)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
        return names

    def _is_scalar_call_name(self, name: str) -> bool:
        """Return True when a bare name call cannot yield tensor data."""
        if name in _SCALAR_BUILTINS or name in self.local_funcs:
            return True
        return name not in self.defined_names

    def _reached_sink(self) -> bool:
        """Return True when the wrapper called its allowed sink."""
        return self.calls_sink if self.sink is not None else self.calls_custom_op

    def _sink_message(self) -> str:
        """Return the missing-sink violation for the active sink."""
        return self.sink.message() if self.sink is not None else (
            "never calls torch.ops.custom_op — the wrapper must call the "
            "evaluator-loaded Ascend C operator"
        )

    def _resolve(
        self, node: ast.AST, unwrap_subscript: bool = False
    ) -> str | None:
        """Return the dotted path for a name or attribute, if resolvable.

        Args:
            node: Expression to resolve.
            unwrap_subscript: Treat kernel_fn[i](...) as a call of
                kernel_fn, which is how a pyasc kernel is launched.
        """
        if unwrap_subscript and isinstance(node, ast.Subscript):
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
        for node in ast.walk(tree):
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
        for node in ast.walk(tree):
            parts = self._assignment_parts(node)
            if parts is None:
                continue
            targets, value = parts
            if len(targets) == 1 and isinstance(targets[0], (ast.Tuple, ast.List)):
                self._bind_unpacked(targets[0].elts, value)
                continue
            for target in targets:
                self._bind_one_target(target, value)

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
        """Treat for-loop range targets as scalar integers."""
        if not (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
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
            return node.id in self.scalar_refs
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
                return self._is_scalar_call_name(base.func.id)
        return isinstance(base, ast.Name) and base.id in self.scalar_refs

    def _is_scalar_call(self, node: ast.Call) -> bool:
        """Return True for builtins, tensor metadata methods, or math.*."""
        func = node.func
        if isinstance(func, ast.Name):
            return self._is_scalar_call_name(func.id)
        if isinstance(func, ast.Attribute):
            if func.attr in _SCALAR_METHODS:
                return True
            path = self._resolve(func)
            return bool(path and path.startswith("math."))
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
        flagged = self._flag_holder_or_dynamic(node)
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
        if name in _DYNAMIC_CALLS:
            self._flag(f"dynamic code execution ({name}())")
            return True
        if name == "getattr" and node.args:
            base = self._resolve(node.args[0])
            if base and (base == "torch" or base.startswith(("torch.", "torch_npu"))):
                self._flag("getattr() on torch modules (dynamic op access)")
                return True
        return flagged

    def _flag_resolved_call(self, func_path: str) -> bool:
        """Flag vendor, functional, and disallowed torch.* calls."""
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
