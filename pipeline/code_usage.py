"""What a piece of Python source does with one library.

Given a file's text and the library's import name, `read_source` reports how the file
imports it, which of its names the file refers to, what it calls with which keyword
arguments, and which methods it calls on the objects it builds from the library's classes.
Names are followed through the file's imports, so `from umap import UMAP as U; U(...)`
and `import umap.umap_ as umap; umap.UMAP(...)` both count as calls to the same class.

The reading is of the syntax tree where the text parses as Python 3. Notebook cells that
do not (shell cells, Python 2) are read with regular expressions instead, which finds the
imports and the calls but follows nothing.

Nothing here looks at what a variable holds when the code runs, so an argument passed as
a variable is reported as "<expr>", and a class reached through a wrapper is not seen.
"""

import ast
import functools
import importlib.util
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


class _Reader:
    """Reads the parsed chunks of one file, which share their imports and variables."""

    def __init__(self, module: str):
        self.module = module
        self.known = library_names(module)
        self.aliases: dict[str, str] = {}  # name in the file -> what it was imported as
        self.instances: dict[str, str] = {}  # variable -> the class it was built from
        self.rebound: set[str] = set()  # variables also assigned something else
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
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    self.aliases[alias.asname or top] = alias.name if alias.asname else top
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                for alias in node.names:
                    if alias.name == "*":
                        self.star = self.star or node.module.split(".")[0] == self.module
                    else:
                        self.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def collect_instances(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign | ast.NamedExpr):
                targets, value = [node.target], node.value
            else:
                continue
            built = self.constructed(value) if isinstance(value, ast.Call) else None
            for target in targets:
                if chain := _chain(target):
                    if built:
                        self.instances[chain] = built
                    else:
                        self.rebound.add(chain)

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
            self.calls.append(
                {
                    "name": name,
                    "args": len(node.args),
                    "kwargs": {k.arg: _literal(k.value) for k in node.keywords if k.arg},
                    "star": any(k.arg is None for k in node.keywords),
                }
            )
        elif name and self.module in name.rsplit(".", 1)[-1].lower():
            # Another library's function or class named after this one: a wrapper around
            # it, or an implementation of its own.
            self.elsewhere.add(name)


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
    pattern = rf"^[ \t]*from[ \t]+({module}[\w.]*)[ \t]+import[ \t]+([^\n#]+)"
    for match in re.finditer(pattern, text, re.MULTILINE):
        for part in match.group(2).strip("() \t").split(","):
            words = part.split()
            if words and words[0].isidentifier():
                heads[words[-1] if "as" in words else words[0]] = f"{match.group(1)}.{words[0]}"
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
            reader.calls.append({"name": name, "args": n_args, "kwargs": kwargs, "star": star})
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
        reader.collect_instances(tree)
    for tree in parsed:
        reader.collect_uses(tree)
    imports = any(reader.ours(target) for target in reader.aliases.values()) or reader.star
    unparsed = [chunk for chunk, tree in zip(chunks, trees, strict=True) if tree is None]
    for chunk in unparsed:
        imports = _read_with_regex(chunk, module, reader) or imports

    return {
        "imports": imports,
        "star_import": reader.star,
        "names": sorted(reader.names),
        "calls": reader.calls,
        "methods": reader.methods,
        "attributes": sorted(reader.attributes),
        "elsewhere": sorted(reader.elsewhere),
        "unparsed": len(unparsed),
    }
