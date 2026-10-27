# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""The type stubs of pysif, held against the compiled module.

pysif is compiled, so an editor's language server cannot read its functions,
their parameters or their docstrings: the stubs in python/pysif/*.pyi say
them instead. The types in them are written by hand -- nothing in the C says
what a function takes or returns -- but everything else is the compiled
module's, and this script keeps it so.

    python packaging/stubs.py check   # what differs; exit status 1 if any
    python packaging/stubs.py fix     # rewrite the stubs' docstrings
    python packaging/stubs.py seed NAME...  # print a stub to start from

check compares every public name of the installed pysif with the stubs: a
name missing on either side, a parameter whose name, kind or default
differs, a property that is a method on the other side, a docstring that
is not the compiled one. tests/python/test_stubs.py runs it.

fix rewrites every docstring in the stubs from the compiled module, so the C
stays the one place the documentation is edited. seed prints a starting
point for a new function, class or module (pysif.io.read_grid, pysif.Field,
pysif.io), with Any for every type.

The installed pysif is what is read: rebuild it (pip install .) after
editing the bindings.
"""

import argparse
import ast
import inspect
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pysif

STUBS = Path(__file__).resolve().parent.parent / "python" / "pysif"

# The modules the stubs describe, and the stub that describes each.
MODULES = {
    "pysif": "__init__.pyi",
    "pysif.io": "io.pyi",
    "pysif.finders": "finders.pyi",
    "pysif.measure": "measure.pyi",
    "pysif.model": "model.pyi",
}


# --- the compiled module ---


@dataclass
class Function:
    signature: inspect.Signature
    doc: str


@dataclass
class Class:
    signature: inspect.Signature | None
    doc: str
    methods: dict[str, Function] = field(default_factory=dict)
    properties: dict[str, str] = field(default_factory=dict)  # name -> doc


def runtime_module(name):
    """The public names of one compiled module: Function, Class, or None
    for anything else (a submodule, pysif.real)."""
    module = sys.modules[name]
    names = {}
    for key, obj in vars(module).items():
        if key.startswith("_"):
            continue
        if isinstance(obj, type):
            names[key] = runtime_class(obj) if obj.__module__.startswith(
                "pysif") else None
        elif callable(obj):
            names[key] = Function(inspect.signature(obj),
                                  inspect.cleandoc(obj.__doc__ or ""))
        else:
            names[key] = None
    return names


def module_doc(name):
    return inspect.cleandoc(sys.modules[name].__doc__ or "")


def runtime_class(cls):
    try:
        signature = inspect.signature(cls)
    except ValueError:
        signature = None
    out = Class(signature, inspect.cleandoc(cls.__doc__ or ""))
    for key, attr in vars(cls).items():
        if key.startswith("_"):
            continue
        doc = inspect.cleandoc(attr.__doc__ or "")
        if isinstance(attr, (types.GetSetDescriptorType,
                             types.MemberDescriptorType)):
            out.properties[key] = doc
        else:
            out.methods[key] = Function(inspect.signature(getattr(cls, key)),
                                        doc)
    return out


# --- the stubs ---


@dataclass
class StubDef:
    """One def in a stub, with what check needs from it. A def under
    @overload is one of several, each a StubDef; a property is the getter."""
    node: ast.FunctionDef
    overload: bool
    is_property: bool


@dataclass
class StubClass:
    node: ast.ClassDef
    defs: dict[str, list[StubDef]]


def stub_module(path):
    """The top-level defs and classes of one stub: name -> list of StubDef,
    or StubClass, or None for a plain assignment (real: ...). The module
    itself is under "", for its docstring."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {"": tree}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            names.setdefault(node.name, []).append(stub_def(node))
        elif isinstance(node, ast.ClassDef):
            defs = {}
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and not is_setter(child):
                    defs.setdefault(child.name, []).append(stub_def(child))
            names[node.name] = StubClass(node, defs)
        elif isinstance(node, (ast.AnnAssign, ast.Assign)):
            target = node.target if isinstance(node, ast.AnnAssign) \
                else node.targets[0]
            if isinstance(target, ast.Name):
                names[target.id] = None
        elif isinstance(node, ast.ImportFrom):
            # "from .measure import X as X" re-exports X.
            for alias in node.names:
                if alias.asname and alias.asname == alias.name:
                    names[alias.name] = None
    return names


def decorators(node):
    return [ast.unparse(d) for d in node.decorator_list]


def is_setter(node):
    return any(d.endswith(".setter") for d in decorators(node))


def stub_def(node):
    names = decorators(node)
    return StubDef(node, "overload" in names or "typing.overload" in names,
                   "property" in names)


def stub_parameters(node, drop_self):
    """(name, kind, default source or None) for each parameter of a def."""
    a = node.args
    positional = a.posonlyargs + a.args
    defaults = [None] * (len(positional) - len(a.defaults)) + a.defaults
    out = []
    for i, (arg, default) in enumerate(zip(positional, defaults)):
        kind = inspect.Parameter.POSITIONAL_ONLY if i < len(a.posonlyargs) \
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        out.append((arg.arg, kind, default))
    if a.vararg:
        out.append((a.vararg.arg, inspect.Parameter.VAR_POSITIONAL, None))
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        out.append((arg.arg, inspect.Parameter.KEYWORD_ONLY, default))
    if a.kwarg:
        out.append((a.kwarg.arg, inspect.Parameter.VAR_KEYWORD, None))
    if drop_self and out and out[0][0] == "self":
        out = out[1:]
    return out


def runtime_parameters(signature):
    return [(p.name, p.kind, p.default)
            for p in signature.parameters.values()]


# --- check ---


class Report:
    def __init__(self):
        self.problems = []

    def add(self, path, node, what, detail):
        root = STUBS.parent.parent
        where = str(path.relative_to(root) if path.is_relative_to(root)
                    else path)
        if node is not None:
            where += f":{getattr(node, 'lineno', 1)}"
        self.problems.append(f"{where}: {what}: {detail}")


def check_parameters(report, path, qualname, sdef, signature,
                     drop_self=False):
    """A def's parameters against the compiled signature. An overload is
    held only to the names: its kinds and defaults are what tell it from
    the others."""
    stub = stub_parameters(sdef.node, drop_self)
    real = runtime_parameters(signature)
    stub_names = [p[0] for p in stub]
    real_names = [p[0] for p in real]
    if stub_names != real_names:
        report.add(path, sdef.node, qualname, "parameters differ "
                   f"(expected {real_names}, found {stub_names})")
        return
    if sdef.overload:
        return
    for (name, kind, default), (_, rkind, rdefault) in zip(stub, real):
        # $self is positional-only in C; stubs write it as a plain self.
        if kind != rkind and name != "self":
            report.add(path, sdef.node, qualname, f"parameter {name}: kind "
                       f"(expected {rkind.description}, found "
                       f"{kind.description})")
        if (default is None) != (rdefault is inspect.Parameter.empty):
            report.add(path, sdef.node, qualname, f"parameter {name}: "
                       "default present on one side only")
        elif default is not None:
            try:
                value = ast.literal_eval(default)
            except ValueError:
                report.add(path, sdef.node, qualname, f"parameter {name}: "
                           f"default is not a literal "
                           f"({ast.unparse(default)})")
                continue
            if repr(value) != repr(rdefault):
                report.add(path, sdef.node, qualname, f"parameter {name}: "
                           f"default (expected {rdefault!r}, found "
                           f"{value!r})")


def check_doc(report, path, qualname, node, doc):
    found = ast.get_docstring(node) or ""
    if found != doc:
        report.add(path, node, qualname,
                   "docstring differs from the compiled one "
                   "(run packaging/stubs.py fix)" if found else
                   "no docstring (run packaging/stubs.py seed for one)")


def check_function(report, path, qualname, defs, fn):
    if any(d.is_property for d in defs):
        report.add(path, defs[0].node, qualname,
                   "a property in the stub, a method in pysif")
        return
    overloads = [d for d in defs if d.overload]
    if overloads and len(overloads) != len(defs):
        report.add(path, defs[0].node, qualname,
                   "overloads and a plain def together")
    elif not overloads and len(defs) > 1:
        report.add(path, defs[1].node, qualname, "defined twice")
    for d in defs:
        check_parameters(report, path, qualname, d, fn.signature)
    # The first overload carries the docstring; editors show it for all.
    check_doc(report, path, qualname, defs[0].node, fn.doc)


def check_class(report, path, qualname, stub, cls):
    check_doc(report, path, qualname, stub.node, cls.doc)
    init = stub.defs.get("__init__")
    if cls.signature is not None:
        if init is None:
            report.add(path, stub.node, qualname, "no __init__")
        else:
            for d in init:
                check_parameters(report, path, qualname + ".__init__", d,
                                 cls.signature, drop_self=True)
    for key, fn in cls.methods.items():
        if key not in stub.defs:
            report.add(path, stub.node, f"{qualname}.{key}",
                       "method missing from the stub")
            continue
        check_function(report, path, f"{qualname}.{key}", stub.defs[key], fn)
    for key, doc in cls.properties.items():
        defs = stub.defs.get(key)
        if defs is None:
            report.add(path, stub.node, f"{qualname}.{key}",
                       "property missing from the stub")
        elif not defs[0].is_property:
            report.add(path, defs[0].node, f"{qualname}.{key}",
                       "a method in the stub, a property in pysif")
        else:
            check_doc(report, path, f"{qualname}.{key}", defs[0].node, doc)
    for key, defs in stub.defs.items():
        if key.startswith("_"):
            continue
        if key not in cls.methods and key not in cls.properties:
            report.add(path, defs[0].node, f"{qualname}.{key}",
                       "in the stub, not in pysif")


def check(stubs=STUBS):
    report = Report()
    for module, filename in MODULES.items():
        path = stubs / filename
        if not path.exists():
            report.add(path, None, module, "no stub")
            continue
        real = runtime_module(module)
        stub = stub_module(path)
        check_doc(report, path, module, stub[""], module_doc(module))
        for key, obj in real.items():
            qualname = f"{module}.{key}"
            if key not in stub:
                report.add(path, None, qualname, "missing from the stub")
            elif isinstance(obj, Function):
                if not isinstance(stub[key], list):
                    report.add(path, None, qualname,
                               "a function in pysif, not in the stub")
                else:
                    check_function(report, path, qualname, stub[key], obj)
            elif isinstance(obj, Class):
                if not isinstance(stub[key], StubClass):
                    report.add(path, None, qualname,
                               "a class in pysif, not in the stub")
                else:
                    check_class(report, path, qualname, stub[key], obj)
        for key in stub:
            if key and not key.startswith("_") and key not in real:
                node = stub[key][0].node if isinstance(stub[key], list) \
                    else getattr(stub[key], "node", None)
                report.add(path, node, f"{module}.{key}",
                           "in the stub, not in pysif")
    return report.problems


# --- fix ---


def render_doc(doc, indent):
    """A docstring as it is written in a stub, at the given indentation."""
    pad = " " * indent
    lines = doc.splitlines()
    if len(lines) == 1:
        return f'{pad}"""{doc}"""'
    body = "\n".join(pad + line if line else "" for line in lines)
    return f'{pad}"""{body.lstrip()}\n{pad}"""'


def doc_targets(module, stub):
    """(docstring node, compiled docstring) for every documented object."""
    yield stub[""], module_doc(module)
    real = runtime_module(module)
    for key, obj in real.items():
        target = stub.get(key)
        if isinstance(obj, Function) and isinstance(target, list):
            yield target[0].node, obj.doc
        elif isinstance(obj, Class) and isinstance(target, StubClass):
            yield target.node, obj.doc
            for name, fn in obj.methods.items():
                if name in target.defs:
                    yield target.defs[name][0].node, fn.doc
            for name, doc in obj.properties.items():
                if name in target.defs:
                    yield target.defs[name][0].node, doc


def fix(stubs=STUBS):
    """Rewrite the docstrings of the stubs; the names of the files changed."""
    changed = []
    for module, filename in MODULES.items():
        path = stubs / filename
        if not path.exists():
            continue
        text = path.read_text()
        lines = text.splitlines(keepends=True)
        edits = []  # (first line, last line, replacement), 0-based
        for node, doc in doc_targets(module, stub_module(path)):
            first = node.body[0]
            has_doc = isinstance(first, ast.Expr) and isinstance(
                getattr(first, "value", None), ast.Constant) and isinstance(
                first.value.value, str)
            if not has_doc or ast.get_docstring(node) == doc:
                continue
            edits.append((first.lineno - 1, first.end_lineno - 1,
                          render_doc(doc, first.col_offset) + "\n"))
        for start, end, new in sorted(edits, reverse=True):
            lines[start:end + 1] = [new]
        if edits:
            path.write_text("".join(lines))
            changed.append(path)
    return changed


# --- seed ---


def render_signature(signature, init=False):
    params = ["self"] if init else []
    star = False
    slash = False
    for p in signature.parameters.values():
        if p.name == "self":  # $self, positional-only in C
            params.append("self")
            continue
        if p.kind == p.POSITIONAL_ONLY:
            slash = True
        elif slash:
            params.append("/")
            slash = False
        if p.kind == p.KEYWORD_ONLY and not star:
            params.append("*")
            star = True
        text = p.name + ": Any"
        if p.kind == p.VAR_POSITIONAL:
            text, star = "*" + text, True
        elif p.kind == p.VAR_KEYWORD:
            text = "**" + text
        if p.default is not p.empty:
            text += f" = {p.default!r}"
        params.append(text)
    if slash:
        params.append("/")
    return ", ".join(params)


def seed_function(name, fn, indent=0):
    pad = " " * indent
    return (f"{pad}def {name}({render_signature(fn.signature)})"
            f" -> Any:\n{render_doc(fn.doc, indent + 4)}\n")


def seed_class(name, cls):
    out = [f"class {name}:\n{render_doc(cls.doc, 4)}\n"]
    if cls.signature is not None:
        out.append(f"    def __init__("
                   f"{render_signature(cls.signature, True)}) -> None: ...\n")
    for key, doc in cls.properties.items():
        out.append(f"    @property\n    def {key}(self) -> Any:\n"
                   f"{render_doc(doc, 8)}\n")
    for key, fn in cls.methods.items():
        out.append(seed_function(key, fn, indent=4))
    return "\n".join(out)


def seed(target):
    """A stub for a module, or one of its functions or classes."""
    if target in MODULES:
        parts = []
        for key, obj in runtime_module(target).items():
            if isinstance(obj, Function):
                parts.append(seed_function(key, obj))
            elif isinstance(obj, Class):
                parts.append(seed_class(key, obj))
        return "from typing import Any\n\n" + "\n".join(parts)
    module, _, key = target.rpartition(".")
    if module not in MODULES:
        raise SystemExit(f"{target}: not a module, function or class of "
                         "pysif (a method comes with its class)")
    obj = runtime_module(module).get(key)
    if isinstance(obj, Function):
        return seed_function(key, obj)
    if isinstance(obj, Class):
        return seed_class(key, obj)
    raise SystemExit(f"{target}: not a function or class of pysif")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="list what differs from the compiled pysif")
    sub.add_parser("fix", help="rewrite the docstrings from the compiled pysif")
    p = sub.add_parser("seed", help="print a stub to start from")
    p.add_argument("names", nargs="+", metavar="NAME",
                   help="a module (pysif.io), function or class")
    args = parser.parse_args()

    if args.command == "check":
        problems = check()
        for line in problems:
            print(line)
        return 1 if problems else 0
    if args.command == "fix":
        for path in fix():
            print(f"{path}: docstrings rewritten")
        return 0
    print("\n\n".join(seed(name) for name in args.names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
