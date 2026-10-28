#!/usr/bin/env python3
"""Check the runners in `harness/` before they are asked to measure anything.

WHAT THIS IS FOR. A runner does not measure anything itself: it writes a
small python program into a string, hands it to a child process, and reads
the child's output. There are TWO programs in a runner and BOTH can fail on
a name that is never bound:

* the CHILD script, in the string;
* the RUNNER MODULE itself.

A name that is used and never bound is a NameError at RUN time, inside a
measurement, on the host, after the quiet window has been waited for. It has
happened in both places: five runners had two lines added to their CHILD
scripts, where the children use `os` without importing it, so every child
died with a NameError and the scale scan produced a table of fifteen rows in
which our nine were CRASH; and the parents had the same mistake and died on
`bench_core.measurement_lock` and `bench_core.with_foreign`. The second kind
is what this file now checks as well. A runner that has been broken since a
merge can sit in the tree for weeks without anybody noticing, because its
table simply stops appearing.

The check is static: no numpy, no site directory, no device, a second to
run, and it runs BEFORE a measurement rather than after it.

USAGE
    python3 tools/smoke_runners.py
    python3 tools/smoke_runners.py harness/scale_scan.py
"""
from __future__ import annotations

import ast
import builtins
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(os.path.dirname(HERE), "harness")

BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__"}


def _bind(target, names):
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _bind(elt, names)
    elif isinstance(target, ast.Starred):
        _bind(target.value, names)


def _scope_bindings(node):
    """The names a scope binds, ignoring its nested scopes.

    The walk is RECURSIVE within the scope: an assignment inside a loop, a
    `with` block, a `try` or an `if` binds the name in that scope, and a
    check that only looked at the direct children of the scope missed every
    one of them.
    """
    names = set()

    def visit(scope):
        for child in ast.iter_child_nodes(scope):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # A nested scope has its own names; only its name belongs here.
                names.add(child.name)
                # A class body's names are visible to its methods, so they are
                # collected too -- which keeps the check quiet about a method
                # that uses a class-level constant.
                visit(child)
                continue
            if isinstance(child, ast.Lambda):
                continue
            if isinstance(child, ast.arguments):
                for a in (list(child.posonlyargs) + list(child.args) + list(child.kwonlyargs)):
                    names.add(a.arg)
                if child.vararg:
                    names.add(child.vararg.arg)
                if child.kwarg:
                    names.add(child.kwarg.arg)
                continue
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    names.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(child, ast.Assign):
                for t in child.targets:
                    _bind(t, names)
            elif isinstance(child, (ast.AugAssign, ast.AnnAssign)):
                _bind(child.target, names)
            elif isinstance(child, (ast.For, ast.AsyncFor)):
                _bind(child.target, names)
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                for item in child.items:
                    if item.optional_vars is not None:
                        _bind(item.optional_vars, names)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                names.add(child.name)
            elif isinstance(child, ast.NamedExpr):
                _bind(child.target, names)
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                for gen in child.generators:
                    _bind(gen.target, names)
            visit(child)

    visit(node)
    return names


def _loads(node, skip_nested_scopes=True):
    """The names a scope reads, ignoring its nested scopes."""
    used = set()
    for child in ast.iter_child_nodes(node):
        if skip_nested_scopes and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                    ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            used.add(child.id)
        used |= _loads(child, skip_nested_scopes)
    return used


def undefined_names(tree, label):
    """The names a program reads and never binds, module level included."""
    problems = []
    # Every name bound anywhere in the program. A name bound inside a
    # function is not really a module-level binding, but treating it as one
    # only makes the check quieter, and a quiet check that finds the real
    # bugs is better than a loud one that drowns them.
    module_bound = _scope_bindings(tree)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = _scope_bindings(node)
        # A name used in a function is bound if the function binds it, or if
        # an enclosing scope binds it (a closure), or if the module binds it.
        free = _loads(node) - local
        for name in sorted(free - module_bound - BUILTINS):
            problems.append("%s: uses %r but never binds it" % (label, name))
    # The module's own body: a name used there and bound nowhere in the
    # module is a NameError when the module runs.
    module_free = _loads(tree) - module_bound - BUILTINS
    for name in sorted(module_free):
        problems.append("%s: the module body uses %r but never binds it" % (label, name))
    return problems


def child_scripts(tree):
    """The embedded child programs of a runner."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        text = node.value.value
        if "sys.argv" not in text:
            continue
        name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else "child"
        out.append((name, text))
    return out


HARNESS_MODULES = ("bench_core", "foreign_cpu")


def check_file(path):
    problems = []
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src, filename=path)
    base = os.path.basename(path)
    # The runner module itself.
    problems += undefined_names(tree, base)
    # The child scripts inside it.
    for name, text in child_scripts(tree):
        try:
            child = ast.parse(text, filename="%s:%s" % (base, name))
        except SyntaxError as exc:
            problems.append("%s: the child script does not parse: %s" % (name, exc))
            continue
        problems += undefined_names(child, "%s:%s" % (base, name))
        # A child must not import the harness's own modules. The child's
        # sys.path holds the built wheel, not the harness directory, so the
        # import cannot be resolved and the child dies on its first line.
        # Three runners carried that line and none of them had produced a
        # table since it was added.
        for node in ast.walk(child):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in HARNESS_MODULES:
                        problems.append(
                            "%s: the child imports %r, which it cannot resolve: "
                            "the child's sys.path holds the built wheel, not the "
                            "harness directory" % (name, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in HARNESS_MODULES:
                    problems.append(
                        "%s: the child imports %r, which it cannot resolve: "
                        "the child's sys.path holds the built wheel, not the "
                        "harness directory" % (name, node.module))
    return problems


def main() -> int:
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = sorted(os.path.join(HARNESS, f) for f in os.listdir(HARNESS)
                       if f.endswith(".py"))
    bad = 0
    for path in files:
        if not os.path.exists(path):
            print("MISSING %s" % path)
            bad += 1
            continue
        problems = check_file(path)
        if problems:
            bad += 1
            for p in problems:
                print("FAIL %s" % p)
        else:
            print("ok   %s" % os.path.basename(path))
    if bad:
        print("\nFAIL: %d runner(s) cannot run as they are" % bad)
        return 1
    print("\nPASS: every runner and every child script binds every name it uses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
