"""What a piece of Python source does with one library.

Given a file's text and the library's import name, `read_source` reports how the file
imports it, which of its names the file refers to, what it calls with which keyword
arguments, and which methods it calls on the objects it builds from the library's classes.
Names are followed through the file's imports, so `from umap import UMAP as U; U(...)`
and `import umap.umap_ as umap; umap.UMAP(...)` both count as calls to the same class.

The reading is of the syntax tree where the text parses as Python 3. Notebook cells that
do not (shell cells, Python 2) are read with regular expressions instead, which finds the
imports and the calls but follows nothing.

Nothing here runs the code. An argument passed as a variable is followed to a value only
where the file gives that variable one constant and nothing else: a single assignment, a
function parameter's default, a command-line option's default, or an entry in a dict
that is unpacked into the call. Such a value is what the code uses unless a caller says
otherwise, and it is marked as followed. Anything else is reported as "<expr>", and a
class reached through a wrapper is not seen.
"""

import ast
import functools
import importlib.util
import json
import re
import warnings
from pathlib import Path

# How stage 05 joins a notebook's code cells into one text.
CELL_BREAK = "\n# %%\n"
EXPRESSION = "<expr>"  # an argument whose value is not written out as a literal
# IPython magics and shell escapes are not Python; each becomes a `pass` at its indent.
MAGIC = re.compile(r"^([ \t]*)[%!].*$", re.MULTILINE)
IMPORT = re.compile(
    r"^[ \t]*(?:from[ \t]+([A-Za-z_]\w*)[\w.]*[ \t]+import\b|import[ \t]+(.+))", re.MULTILINE
)
# The scikit-learn estimator interface. A variable can hold the library under one of these
# names and an object built from it under the same name (`umap = umap.UMAP()`), and these
# are the attributes that then belong to the object.
ESTIMATOR_ATTRS = {
    "fit",
    "fit_transform",
    "transform",
    "inverse_transform",
    "update",
    "set_params",
    "get_params",
    "embedding_",
    "graph_",
}
MAX_CALL_CHARS = 3000
# How many variables a value is followed through: `self.k = k`, where k has a default.
MAX_FOLLOW = 3
IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


@functools.cache
def library_names(module: str) -> frozenset[str] | None:
    """The submodules and top-level names of the installed library, read without importing it.

    A file that imports the library and then reuses its name for a variable
    (`umap = reducer.fit_transform(X)`) goes on to write `umap.max()`. Knowing what the
    library really has tells those apart. None if the library is not installed.
    """
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    origin = Path(spec.origin)
    if origin.name != "__init__.py":
        return None
    names = {path.stem for path in origin.parent.glob("*.py")}
    names |= {path.name for path in origin.parent.iterdir() if (path / "__init__.py").exists()}
    for node in ast.walk(ast.parse(origin.read_text())):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names |= {alias.asname or alias.name for alias in node.names}
        elif isinstance(node, ast.Assign):
            names |= {target.id for target in node.targets if isinstance(target, ast.Name)}
    return frozenset(names)


def imported_modules(text: str) -> set[str]:
    """The top-level modules a text imports, read from its import lines alone."""
    modules = set()
    for match in IMPORT.finditer(text):
        if match.group(1):
            modules.add(match.group(1))
            continue
        for part in match.group(2).split("#")[0].split(","):
            words = part.split()
            if words and (top := words[0].split(".")[0]).isidentifier():
                modules.add(top)
    return modules


def _parse(text: str) -> ast.AST | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # invalid escape sequences in other people's strings
        for candidate in (text, MAGIC.sub(r"\1pass", text)):
            try:
                return ast.parse(candidate)
            except (SyntaxError, ValueError, RecursionError, MemoryError):
                continue
    return None


def _chain(node: ast.AST) -> str | None:
    """`a.b.c` for a name or a run of attribute accesses, as written."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def _literal(node: ast.AST):
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return EXPRESSION
    if isinstance(value, float) and value != value:  # NaN does not survive JSON
        return EXPRESSION
    return value if isinstance(value, str | int | float | bool | type(None)) else EXPRESSION


@functools.cache
def class_signature(module: str, class_name: str) -> tuple[tuple[str, ...], dict] | None:
    """A library class's constructor parameters, in order, and their constant defaults.

    Read from the installed library's source without importing it, so that positional
    arguments can be given their names and an argument left out can be given its value.
    None if the library is not installed or has no such class.
    """
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    for path in sorted(Path(spec.origin).parent.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.ClassDef) and node.name == class_name):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    names = [arg.arg for arg in item.args.args[1:]]
                    defaults = [_literal(default) for default in item.args.defaults]
                    with_defaults = names[len(names) - len(defaults) :]
                    return tuple(names), dict(zip(with_defaults, defaults, strict=True))
    return None


class _Reader:
    """Reads the parsed chunks of one file, which share their imports and variables."""

    def __init__(self, module: str):
        self.module = module
        self.known = library_names(module)
        self.aliases: dict[str, str] = {}  # name in the file -> what it was imported as
        self.instances: dict[str, str] = {}  # variable -> the class it was built from
        self.rebound: set[str] = set()  # variables also assigned something else
        # Every value the file gives a variable; None stands for one it does not show,
        # such as a parameter with no default or a loop variable.
        self.assigned: dict[str, list[ast.AST | None]] = {}
        self.options: dict[str, list[ast.AST]] = {}  # command-line option -> its defaults
        self.import_forms: set[str] = set()
        self.guarded = False  # imported inside a try that catches a failed import
        self.lazy = False  # imported inside a function
        self.star = False
        self.names: set[str] = set()
        self.calls: list[dict] = []
        self.methods: list[dict] = []
        self.attributes: set[str] = set()
        self.elsewhere: set[str] = set()

    def ours(self, name: str | None) -> bool:
        """Whether a dotted name is one of the library's."""
        if not name:
            return False
        parts = name.split(".")
        if parts[0] != self.module:
            return False
        return self.known is None or len(parts) == 1 or parts[1] in self.known

    def resolve(self, node: ast.AST) -> str | None:
        """The dotted name a reference stands for, through the file's imports."""
        chain = _chain(node)
        if chain is None:
            return None
        head, _, rest = chain.partition(".")
        if head not in self.aliases:
            return None
        return self.aliases[head] + (f".{rest}" if rest else "")

    def on_instance(self, node: ast.AST) -> str | None:
        """The class behind `x.attr` when x holds an object built from the library."""
        if not isinstance(node, ast.Attribute):
            return None
        if isinstance(node.value, ast.Call):
            return self.constructed(node.value)
        owner = _chain(node.value)
        if owner not in self.instances:
            return None
        # A name that also holds the library, or something else, is taken for the object
        # only where an estimator's own attribute is used.
        shared = owner in self.aliases or owner in self.rebound
        if shared and node.attr not in ESTIMATOR_ATTRS:
            return None
        return self.instances[owner]

    def constructed(self, call: ast.Call) -> str | None:
        """The library class a call builds an object of, if it does."""
        name = self.resolve(call.func)
        if self.ours(name) and name.rsplit(".", 1)[-1][:1].isupper():
            return name
        # fit() returns the estimator, so `UMAP().fit(X)` and `model.fit(X)` are still the object.
        if isinstance(call.func, ast.Attribute) and call.func.attr == "fit":
            if isinstance(call.func.value, ast.Call):
                return self.constructed(call.func.value)
            return self.instances.get(_chain(call.func.value))
        return None

    def collect_imports(self, tree: ast.AST) -> None:
        nodes = list(ast.walk(tree))
        parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    self.aliases[alias.asname or top] = alias.name if alias.asname else top
                    if top == self.module:
                        # The one alias worth keeping: a submodule under the library's name.
                        renamed = f" as {alias.asname}" if alias.asname == self.module else ""
                        dotted = alias.name != self.module
                        self.note_import(
                            node, f"import {alias.name}{renamed if dotted else ''}", parents
                        )
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                ours = node.module.split(".")[0] == self.module
                for alias in node.names:
                    if alias.name == "*":
                        self.star = self.star or ours
                    else:
                        self.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
                    if ours:
                        self.note_import(node, f"from {node.module} import {alias.name}", parents)

    def note_import(self, node: ast.AST, form: str, parents: dict) -> None:
        """Record how the library is imported, and whether warily or late."""
        self.import_forms.add(form)
        child, parent = node, parents.get(node)
        while parent is not None:
            if isinstance(parent, ast.Try) and child in parent.body:
                caught = {
                    name.id if isinstance(name, ast.Name) else getattr(name, "attr", "")
                    for handler in parent.handlers
                    for name in (
                        handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
                    )
                    if name is not None
                }
                bare = any(handler.type is None for handler in parent.handlers)
                self.guarded = self.guarded or bare or bool(caught & IMPORT_ERRORS)
            elif isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
                self.lazy = True
            child, parent = parent, parents.get(parent)

    def collect_assignments(self, tree: ast.AST) -> None:
        """Note what each variable is given: the objects built from the library, and the
        values a later argument can be followed to."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                self.note_parameters(node.args)
            elif isinstance(node, ast.Call):
                self.note_option(node)
            elif isinstance(node, ast.AugAssign | ast.For | ast.AsyncFor | ast.comprehension):
                self.note_values(node.target, None)
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                self.note_values(node.optional_vars, None)

            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign | ast.NamedExpr) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            built = self.constructed(value) if isinstance(value, ast.Call) else None
            for target in targets:
                self.note_values(target, value)
                if chain := _chain(target):
                    if built:
                        self.instances[chain] = built
                    else:
                        self.rebound.add(chain)

    def note_values(self, target: ast.AST, value: ast.AST | None) -> None:
        if isinstance(target, ast.Tuple | ast.List):
            for element in target.elts:  # unpacked: which value goes where is not followed
                self.note_values(element, None)
        elif chain := _chain(target):
            self.assigned.setdefault(chain, []).append(value)

    def note_parameters(self, args: ast.arguments) -> None:
        positional = [*args.posonlyargs, *args.args]
        defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
        pairs = [*zip(positional, defaults, strict=True)]
        pairs += zip(args.kwonlyargs, args.kw_defaults, strict=True)
        pairs += [(arg, None) for arg in (args.vararg, args.kwarg) if arg is not None]
        for arg, default in pairs:
            self.assigned.setdefault(arg.arg, []).append(default)

    def note_option(self, node: ast.Call) -> None:
        """Record the default of an argparse or click option: `args.k` is read from it."""
        if not isinstance(node.func, ast.Attribute):
            return
        if node.func.attr not in ("add_argument", "option", "add_option"):
            return
        keywords = {k.arg: k.value for k in node.keywords if k.arg}
        if "default" not in keywords:
            return
        flags = [_literal(arg) for arg in node.args]
        flags = [f for f in flags if isinstance(f, str) and f.startswith("-")]
        dest = _literal(keywords["dest"]) if "dest" in keywords else None
        if not isinstance(dest, str):
            if not flags:
                return
            dest = max(flags, key=len).lstrip("-").replace("-", "_")
        self.options.setdefault(dest, []).append(keywords["default"])

    def value_of(self, node: ast.AST, depth: int = 0) -> tuple[object, bool]:
        """A node's constant value, and whether it was followed through a variable."""
        value = _literal(node)
        if value != EXPRESSION:
            return value, depth > 0
        chain = _chain(node)
        if chain is None or depth >= MAX_FOLLOW:
            return EXPRESSION, False
        sources = self.assigned.get(chain)
        if sources is None and isinstance(node, ast.Attribute):
            sources = self.options.get(node.attr)  # args.n_neighbors, from --n_neighbors
        if not sources or any(source is None for source in sources):
            return EXPRESSION, False
        found = {json.dumps(self.value_of(source, depth + 1)[0]) for source in sources}
        if len(found) != 1 or json.loads(next(iter(found))) == EXPRESSION:
            return EXPRESSION, False
        return json.loads(found.pop()), True

    def unpacked(self, node: ast.AST) -> tuple[list[tuple[str, ast.AST]], bool] | None:
        """The entries of a dict unpacked into a call with `**`, and whether it was a variable."""
        via_variable = False
        if chain := _chain(node):
            sources = self.assigned.get(chain)
            if not sources or len(sources) != 1 or sources[0] is None:
                return None
            node, via_variable = sources[0], True
        if isinstance(node, ast.Dict):
            keys = [_literal(key) if key is not None else None for key in node.keys]
            if all(isinstance(key, str) for key in keys):
                return list(zip(keys, node.values, strict=True)), via_variable
        elif isinstance(node, ast.Call) and _chain(node.func) == "dict" and not node.args:
            if all(k.arg for k in node.keywords):
                return [(k.arg, k.value) for k in node.keywords], via_variable
        return None

    def collect_uses(self, tree: ast.AST) -> None:
        nodes = list(ast.walk(tree))
        # The inner links of `a.b.c` are not references of their own.
        inner = {id(n.value) for n in nodes if isinstance(n, ast.Attribute)}
        called = {id(n.func) for n in nodes if isinstance(n, ast.Call)}
        for node in nodes:
            if isinstance(node, ast.Call):
                self.read_call(node)
            if id(node) in inner or not isinstance(node, ast.Name | ast.Attribute):
                continue
            if not isinstance(node.ctx, ast.Load):
                continue
            if built := self.on_instance(node):
                if id(node) not in called:  # a method call is recorded as one, not as a read
                    self.attributes.add(f"{built}.{node.attr}")
            elif self.ours(name := self.resolve(node)):
                self.names.add(name)

    def read_call(self, node: ast.Call) -> None:
        if built := self.on_instance(node.func):
            self.methods.append(
                {
                    "on": built,
                    "method": node.func.attr,
                    "args": len(node.args),
                    "kwargs": sorted(k.arg for k in node.keywords if k.arg),
                    # The variable is assigned something else somewhere in the file, so
                    # this may be a call on that other thing.
                    "uncertain": _chain(node.func.value) in self.rebound,
                }
            )
            return
        name = self.resolve(node.func)
        if self.ours(name):
            self.calls.append(self.read_arguments(name, node))
        elif name and self.module in name.rsplit(".", 1)[-1].lower():
            # Another library's function or class named after this one: a wrapper around
            # it, or an implementation of its own.
            self.elsewhere.add(name)

    def read_arguments(self, name: str, node: ast.Call) -> dict:
        """One call to the library: each argument's name and, where it can be had, its value."""
        last = name.rsplit(".", 1)[-1]
        signature = class_signature(self.module, last) if last[:1].isupper() else None
        given: list[tuple[str, ast.AST]] = []
        for parameter, arg in zip(signature[0] if signature else (), node.args, strict=False):
            if isinstance(arg, ast.Starred):
                break
            given.append((parameter, arg))
        given += [(k.arg, k.value) for k in node.keywords if k.arg]

        arguments, followed, star = {}, set(), False
        for key, value_node in given:
            arguments[key], was_followed = self.value_of(value_node)
            if was_followed:
                followed.add(key)
        for keyword in node.keywords:
            if keyword.arg is not None:
                continue
            entries = self.unpacked(keyword.value)
            if entries is None:
                star = True  # arguments this reading cannot see
                continue
            for key, value_node in entries[0]:
                if key in arguments:
                    continue
                arguments[key], was_followed = self.value_of(value_node)
                if arguments[key] != EXPRESSION and (was_followed or entries[1]):
                    followed.add(key)
        return {
            "name": name,
            "args": len(node.args),
            "kwargs": arguments,
            "followed": sorted(followed),
            "star": star,
        }


def _split_arguments(text: str) -> list[str]:
    """Split a call's argument text at its top-level commas."""
    parts, depth, start = [], 0, 0
    for i, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def _read_with_regex(text: str, module: str, reader: _Reader) -> bool:
    """Find imports of the library and calls through them in text that does not parse.

    Returns whether the text imports the library.
    """
    heads: dict[str, str] = {}
    pattern = rf"^[ \t]*import[ \t]+({module}[\w.]*)(?:[ \t]+as[ \t]+(\w+))?"
    for match in re.finditer(pattern, text, re.MULTILINE):
        heads[match.group(2) or module] = match.group(1) if match.group(2) else module
        renamed = f" as {module}" if match.group(2) == module and match.group(1) != module else ""
        reader.import_forms.add(f"import {match.group(1)}{renamed}")
    pattern = rf"^[ \t]*from[ \t]+({module}[\w.]*)[ \t]+import[ \t]+([^\n#]+)"
    for match in re.finditer(pattern, text, re.MULTILINE):
        for part in match.group(2).strip("() \t").split(","):
            words = part.split()
            if words and words[0].isidentifier():
                heads[words[-1] if "as" in words else words[0]] = f"{match.group(1)}.{words[0]}"
                reader.import_forms.add(f"from {match.group(1)} import {words[0]}")
    for head, resolved in heads.items():
        call = rf"(?<![\w.]){re.escape(head)}((?:\.\w+)*)[ \t]*\("
        for match in re.finditer(call, text):
            name = resolved + match.group(1)
            depth, end = 1, match.end()
            while depth and end < min(len(text), match.end() + MAX_CALL_CHARS):
                depth += (text[end] in "([{") - (text[end] in ")]}")
                end += 1
            if depth:
                continue
            kwargs, n_args, star = {}, 0, False
            for part in _split_arguments(text[match.end() : end - 1]):
                keyword = re.match(r"(\w+)[ \t]*=(?!=)(.*)", part, re.DOTALL)
                if part.startswith("**"):
                    star = True
                elif keyword:
                    try:
                        value = _literal(ast.parse(keyword.group(2).strip(), mode="eval").body)
                    except (SyntaxError, ValueError, RecursionError, MemoryError):
                        value = EXPRESSION
                    kwargs[keyword.group(1)] = value
                else:
                    n_args += 1
            reader.names.add(name)
            reader.calls.append(
                {"name": name, "args": n_args, "kwargs": kwargs, "followed": [], "star": star}
            )
    return bool(heads)


def read_source(text: str, module: str) -> dict:
    """What one file's source does with `module`."""
    reader = _Reader(module)
    whole = _parse(text)
    chunks = [text] if whole is not None else text.split(CELL_BREAK)
    trees = [whole] if whole is not None else [_parse(chunk) for chunk in chunks]

    parsed = [tree for tree in trees if tree is not None]
    for tree in parsed:
        reader.collect_imports(tree)
    for tree in parsed:
        reader.collect_assignments(tree)
    for tree in parsed:
        reader.collect_uses(tree)
    imports = any(reader.ours(target) for target in reader.aliases.values()) or reader.star
    unparsed = [chunk for chunk, tree in zip(chunks, trees, strict=True) if tree is None]
    for chunk in unparsed:
        imports = _read_with_regex(chunk, module, reader) or imports

    return {
        "imports": imports,
        "star_import": reader.star,
        "import_forms": sorted(reader.import_forms),
        "guarded": reader.guarded,
        "lazy": reader.lazy,
        "names": sorted(reader.names),
        "calls": reader.calls,
        "methods": reader.methods,
        "attributes": sorted(reader.attributes),
        "elsewhere": sorted(reader.elsewhere),
        "unparsed": len(unparsed),
    }
