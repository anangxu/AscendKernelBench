"""AST-level semantic checks for model_new.py."""

from __future__ import annotations

import ast

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


################################ AST VISITOR #################################
class WrapperSemantics(ast.NodeVisitor):
    """AST checks for model_new.py."""

    def __init__(self, source: str) -> None:
        """Parse source and populate violations."""
        self.violations: list[str] = []
        self.aliases: dict[str, str] = {}
        self.co_refs: set[str] = set()
        self.holders: set[str] = set()
        self.scalar_refs: set[str] = set()
        self.tensor_refs: set[str] = set()
        self.calls_custom_op = False
        tree = ast.parse(source)
        self._collect_aliases(tree)
        self._collect_forward_params(tree)
        self._collect_bindings_to_fixpoint(tree)
        self.visit(tree)
        if not self.calls_custom_op:
            self.violations.append(
                "never calls torch.ops.custom_op — the wrapper must call the "
                "evaluator-loaded Ascend C operator"
            )

    def _resolve(self, node: ast.AST) -> str | None:
        """Return the dotted path for a name or attribute, if resolvable."""
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
        """Return True if path is torch.ops.custom_op or an alias."""
        if path == "torch.ops.custom_op" or path.startswith("torch.ops.custom_op."):
            return True
        return any(path == ref or path.startswith(ref + ".") for ref in self.co_refs)

    def _collect_aliases(self, tree: ast.AST) -> None:
        """Record import aliases and flag banned import roots."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self._record_import(node)
            elif isinstance(node, ast.ImportFrom):
                self._record_import_from(node)

    def _collect_forward_params(self, tree: ast.AST) -> None:
        """Record forward/__call__ parameters as likely tensor names."""
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in {"forward", "__call__"}:
                continue
            args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            for arg in args:
                if arg.arg != "self":
                    self.tensor_refs.add(arg.arg)
            if node.args.vararg:
                self.tensor_refs.add(node.args.vararg.arg)
            if node.args.kwarg:
                self.tensor_refs.add(node.args.kwarg.arg)

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

    def _collect_bindings_to_fixpoint(self, tree: ast.AST) -> None:
        """Repeat binding propagation until no new name is classified."""
        while True:
            state = (len(self.scalar_refs), len(self.co_refs), len(self.holders))
            self._collect_bindings(tree)
            if state == (
                len(self.scalar_refs),
                len(self.co_refs),
                len(self.holders),
            ):
                return

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
        if (
            isinstance(base, ast.Call)
            and isinstance(base.func, ast.Attribute)
            and base.func.attr == "size"
        ):
            return True
        return isinstance(base, ast.Name) and base.id in self.scalar_refs

    def _is_scalar_call(self, node: ast.Call) -> bool:
        """Return True for builtins, tensor metadata methods, or math.*."""
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in _SCALAR_BUILTINS:
                return True
            if func.id in _SCALAR_IF_ARGUMENTS_ARE_SCALAR:
                return not any(self._is_known_tensor(arg) for arg in node.args)
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
        """Allow torch.ops.custom_op; flag other compute or side effects."""
        func_path = self._resolve(node.func)
        if func_path and self._is_custom_op_path(func_path):
            self.calls_custom_op = True
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
                "compute must live in the Ascend C kernel"
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
            self._flag(f"{func_path}() is not allowed in model_new.py")
            return True
        if func_path.startswith(("os.system", "os.popen", "os.exec", "os.spawn")):
            self._flag(f"host process execution: {func_path}()")
            return True
        return False

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Flag tensor arithmetic and @; allow integer shape math."""
        if isinstance(node.op, ast.MatMult):
            self._flag("@ (matmul) operator — compute must live in the Ascend C kernel")
        elif isinstance(node.op, _ARITH_OPS) and not (
            self._is_scalar(node.left) and self._is_scalar(node.right)
        ):
            self._flag(
                "arithmetic on non-scalar values — tensor compute "
                "must live in the Ascend C kernel (integer shape "
                "arithmetic is fine)"
            )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Flag in-place tensor arithmetic and @=."""
        if not isinstance(node.op, ast.MatMult):
            if not (self._is_scalar(node.target) and self._is_scalar(node.value)):
                self._flag(
                    "in-place arithmetic on non-scalar values — tensor compute "
                    "must live in the Ascend C kernel"
                )
        else:
            self._flag("@ (matmul) operator — compute must live in the Ascend C kernel")
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        """Flag unary arithmetic on non-scalar (tensor) values."""
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
                "compute and belong in the Ascend C kernel"
            )
        self.generic_visit(node)


################################ AST VISITOR #################################
