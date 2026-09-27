"""Dependency stand-ins for the plain-script tests.

The generation layer only needs YAML, pydantic, loguru, and openai at import
time; these factories supply tiny stand-ins so a test can exercise the real
engine modules on a machine with no benchmark dependencies, no torch, and no
NPU. A real package is left untouched when it is importable.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

MISSING = object()


def module_available(name: str) -> bool:
    """Return True when a top-level module can be imported."""
    try:
        __import__(name)
    except Exception:
        return False
    return True


def install_loguru() -> None:
    """Install a no-op loguru module when it is missing."""
    if module_available("loguru"):
        return
    module = types.ModuleType("loguru")

    class _Logger:
        def __getattr__(self, name: str) -> Any:
            return lambda *args, **kwargs: None

    module.logger = _Logger()
    sys.modules["loguru"] = module


def install_openai(fake_client: type) -> None:
    """Point the openai client class at a scripted fake."""
    if module_available("openai"):
        openai = sys.modules["openai"]
    else:
        openai = types.ModuleType("openai")
        sys.modules["openai"] = openai
    openai.OpenAI = fake_client


def install_openai_placeholder() -> None:
    """Install an openai module whose client refuses to run.

    Used by smoke checks that only need the import to succeed; the tests
    install a scripted client instead with install_openai.
    """
    if module_available("openai"):
        return

    class _RefusingClient:
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("no LLM endpoint in this check")

    install_openai(_RefusingClient)


def install_torch() -> None:
    """Install a torch stand-in with real shape arithmetic only."""
    if module_available("torch"):
        return

    class _Tensor:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape
            self.dtype = "float32"
            self.device = "npu:0"

        def numel(self) -> int:
            total = 1
            for size in self.shape:
                total *= size
            return total

        def dim(self) -> int:
            return len(self.shape)

        def size(self, index: int) -> int:
            return self.shape[index]

    def empty_like(tensor: Any) -> _Tensor:
        return _Tensor(tuple(tensor.shape))

    def empty(*shape: Any, **kwargs: Any) -> _Tensor:
        flat: list[int] = []
        for item in shape:
            flat.extend(item if isinstance(item, (tuple, list)) else [item])
        return _Tensor(tuple(int(value) for value in flat))

    module = types.ModuleType("torch")
    module.__path__ = []
    module.Tensor = _Tensor
    module.empty_like = empty_like
    module.empty = empty
    nn = types.ModuleType("torch.nn")

    class _Module:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    nn.Module = _Module
    module.nn = nn
    sys.modules["torch"] = module
    sys.modules["torch.nn"] = nn


def install_rich() -> None:
    """Install the small rich surface the CLIs import."""
    if module_available("rich"):
        return

    class _Console:
        def print(self, *args: Any, **kwargs: Any) -> None:
            return None

    def _install_traceback(*args: Any, **kwargs: Any) -> None:
        return None

    console = types.ModuleType("rich.console")
    console.Console = _Console
    traceback = types.ModuleType("rich.traceback")
    traceback.install = _install_traceback
    rich = types.ModuleType("rich")
    rich.__path__ = []
    rich.console = console
    rich.traceback = traceback
    sys.modules["rich"] = rich
    sys.modules["rich.console"] = console
    sys.modules["rich.traceback"] = traceback


def install_pydantic() -> None:
    """Install a minimal pydantic stand-in (BaseModel/Field/validators)."""
    if module_available("pydantic"):
        return
    import typing

    class _FieldInfo:
        def __init__(self, default: Any = MISSING, **kwargs: Any) -> None:
            self.default = default
            self.kwargs = kwargs

    internals = {
        "model_config",
        "__model_fields__",
        "__field_defaults__",
        "__validators__",
    }

    class _ModelMeta(type):
        def __new__(cls, name: str, bases: tuple, namespace: dict) -> type:
            created = super().__new__(cls, name, bases, namespace)
            fields: dict[str, Any] = {}
            defaults: dict[str, Any] = {}
            validators: dict[str, Any] = {}
            for base in reversed(created.__mro__[1:]):
                fields.update(getattr(base, "__model_fields__", {}))
                defaults.update(getattr(base, "__field_defaults__", {}))
                validators.update(getattr(base, "__validators__", {}))
            fields.update(
                {
                    key: value
                    for key, value in namespace.get("__annotations__", {}).items()
                    if key not in internals
                }
            )
            for key, value in namespace.items():
                if key in internals or key.startswith("_"):
                    continue
                if isinstance(value, _FieldInfo):
                    if value.default is not MISSING:
                        defaults[key] = value.default
                    elif "default_factory" in value.kwargs:
                        defaults[key] = value.kwargs["default_factory"]()
                elif key in fields:
                    defaults[key] = value
                for field_name in getattr(value, "__validator_names__", ()):
                    validators[field_name] = value
            created.__model_fields__ = fields
            created.__field_defaults__ = defaults
            created.__validators__ = validators
            return created

    class _BaseModel(metaclass=_ModelMeta):
        model_config: dict = {}
        __model_fields__: dict = {}
        __field_defaults__: dict = {}
        __validators__: dict = {}

        def __init__(self, **values: Any) -> None:
            hints = _own_hints(type(self))
            for name in self.__model_fields__:
                if name in values:
                    raw = values[name]
                elif name in self.__field_defaults__:
                    raw = self.__field_defaults__[name]
                else:
                    raise TypeError(f"{type(self).__name__}.{name} is required")
                validator = self.__validators__.get(name)
                if validator is not None:
                    raw = validator(type(self), raw)
                object.__setattr__(self, name, _coerce(raw, hints.get(name, Any)))

        def __setattr__(self, name: str, value: Any) -> None:
            if self.model_config.get("frozen"):
                raise TypeError(f"{type(self).__name__} is frozen")
            object.__setattr__(self, name, value)

        @classmethod
        def model_validate(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                raise TypeError("model_validate expects a mapping")
            return cls(**data)

        def model_dump(self) -> dict:
            return {name: getattr(self, name) for name in self.__model_fields__}

    def _own_hints(cls: type) -> dict[str, Any]:
        try:
            hints = typing.get_type_hints(cls)
        except Exception:
            return dict(cls.__model_fields__)
        own = set(cls.__dict__.get("__annotations__", {}))
        return {key: value for key, value in hints.items() if key in own}

    def _coerce(value: Any, annotation: Any) -> Any:
        if annotation is Any or annotation is None:
            return value
        if value is None:
            if annotation is type(None) or type(None) in typing.get_args(annotation):
                return None
            raise TypeError("value may not be None")
        args = typing.get_args(annotation)
        if type(None) in args:
            args = tuple(item for item in args if item is not type(None))
        if len(args) == 1:
            annotation = args[0]
        if annotation is Any:
            return value
        if annotation is bool:
            if not isinstance(value, bool):
                raise TypeError("expected bool")
            return value
        if annotation in (int, float, str):
            return annotation(value)
        return value

    def _field(default: Any = MISSING, **kwargs: Any) -> Any:
        return _FieldInfo(default, **kwargs)

    def _field_validator(*names: str, mode: str = "after") -> Any:
        def _decorator(func: Any) -> Any:
            raw = func.__func__ if isinstance(func, classmethod) else func

            def _wrapped(cls: Any, value: Any) -> Any:
                return raw(cls, value) if mode == "before" else raw(value)

            _wrapped.__validator_names__ = names
            return _wrapped

        return _decorator

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = _BaseModel
    pydantic.ConfigDict = lambda **kwargs: dict(kwargs)
    pydantic.Field = _field
    pydantic.field_validator = _field_validator
    pydantic.ValidationError = ValueError
    sys.modules["pydantic"] = pydantic


def install_yaml() -> None:
    """Install a PyYAML stand-in for the flat generation config subset."""
    if module_available("yaml"):
        return
    module = types.ModuleType("yaml")
    module.YAMLError = ValueError
    module.safe_dump = lambda payload, **kwargs: yaml_dump(payload, 0)
    module.safe_load = yaml_load
    sys.modules["yaml"] = module


def install_engine_stubs() -> None:
    """Install every stand-in the generation engine imports."""
    install_loguru()
    install_yaml()
    install_pydantic()
    install_torch()
    install_rich()
    install_openai_placeholder()


def _yaml_scalar(value: Any) -> str:
    """Render one YAML scalar the way PyYAML renders it."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    if (
        text == ""
        or text.strip() != text
        or text.lower()
        in {
            "true",
            "false",
            "null",
            "none",
            "~",
        }
    ):
        return json.dumps(text)
    try:
        float(text)
    except ValueError:
        return text
    return json.dumps(text)


def yaml_dump(value: Any, indent: int) -> str:
    """Emit the flat mapping / list subset of YAML."""
    pad = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{pad}{key}:")
                lines.append(yaml_dump(item, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {_yaml_scalar(item)}" for item in value)
    return f"{pad}{_yaml_scalar(value)}"


def yaml_load(text: str) -> Any:
    """Parse exactly what yaml_dump emits."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    content = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    for index, line in enumerate(content):
        indent = len(line) - len(line.lstrip(" "))
        entry = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if entry.startswith("- "):
            parent = stack[-1][1]
            if not isinstance(parent, list):
                raise ValueError(f"list item outside a list: {line!r}")
            parent.append(yaml_value(entry[2:]))
            continue
        key, _, rest = entry.partition(":")
        container = stack[-1][1]
        if not isinstance(container, dict):
            raise ValueError(f"mapping key inside a list: {line!r}")
        if rest.strip():
            container[key.strip()] = yaml_value(rest.strip())
            continue
        following = content[index + 1] if index + 1 < len(content) else None
        child: Any = [] if following and following.strip().startswith("- ") else {}
        container[key.strip()] = child
        stack.append((indent, child))
    return root


def yaml_value(token: str) -> Any:
    """Parse one scalar or inline JSON list token."""
    if token.startswith("[") and token.endswith("]"):
        return json.loads(token)
    if token.startswith('"'):
        return json.loads(token)
    if token in {"~", "null", "None"}:
        return None
    if token in {"true", "false"}:
        return token == "true"
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token
