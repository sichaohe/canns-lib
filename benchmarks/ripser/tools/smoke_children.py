#!/usr/bin/env python3
"""Check the CHILD scripts embedded in the runners, before they are measured.

WHAT THIS IS FOR. A runner in `harness/` does not measure anything itself: it
writes a small python program into a string, hands it to a child process, and
reads the child's output. That string is a PROGRAM, and a program with a name
it never imported fails at RUN time -- inside a measurement, on the host,
after the quiet window has been waited for.

That is not hypothetical. A commit added two lines to five child scripts:

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bench_core

The children use `os` and none of them import it, and none of them need
bench_core either. Every one of them died with a NameError on its first line
of work, so the scale scan produced fifteen rows of which nine were CRASHes,
and the quiet-window gate reported the batch as "contaminated" -- a word that
points at the host instead of at the bug.

So this check runs the static half of the job: for every child script in the
harness, it reports the names the script USES but never BINDS. It needs no
numpy, no site directory and no GPU, so it runs in a second, and it runs
before the measurement rather than after it.

A name is "bound" if the script imports it, assigns it, defines it, or takes
it as an argument. A name that is only ever READ and never bound is the bug.

USAGE
    python3 tools/smoke_children.py                 # check every runner
    python3 tools/smoke_children.py harness/scale_scan.py
"""
from __future__ import annotations

import ast
import builtins
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(os.path.dirname(HERE), "harness")

BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__"}


def bound_names(tree):
    """Every name the script binds somewhere, in any scope."""
    names = set()

    def bind(target):
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                bind(elt)
        elif isinstance(target, ast.Starred):
            bind(target.value)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            args = getattr(node, "args", None)
            if args is not None:
                for a in (args.posonlyargs + args.args + args.kwonlyargs):
                    names.add(a.arg)
                if args.vararg:
                    names.add(args.vararg.arg)
                if args.kwarg:
                    names.add(args.kwarg.arg)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                bind(t)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            bind(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            bind(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    bind(item.optional_vars)
    # except handlers and comprehension targets are covered by ast.walk on
    # the specific node types below.
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for gen in node.generators:
                bind(gen.target)
        elif isinstance(node, ast.NamedExpr):
            bind(node.target)
    return names


def used_names(tree):
    """Every name the script reads."""
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
    return used


def child_scripts(path):
    """The embedded child programs of one runner: (name, source)."""
    with open(path) as fh:
        src = fh.read()
    tree = ast.parse(src, filename=path)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        text = node.value.value
        # A child program: it reads its arguments and runs something.
        if "sys.argv" not in text:
            continue
        name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else "child"
        out.append((name, text))
    return out


def check_file(path):
    problems = []
    for name, text in child_scripts(path):
        try:
            tree = ast.parse(text, filename="%s:%s" % (os.path.basename(path), name))
        except SyntaxError as exc:
            problems.append("%s: the child script does not parse: %s" % (name, exc))
            continue
        bound = bound_names(tree)
        used = used_names(tree)
        undefined = sorted(used - bound - BUILTINS)
        for nm in undefined:
            problems.append("%s: the child script uses %r but never binds it" % (name, nm))
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
                print("FAIL %s: %s" % (os.path.basename(path), p))
        else:
            print("ok   %s" % os.path.basename(path))
    if bad:
        print("\nFAIL: %d runner(s) have a child script that cannot run" % bad)
        return 1
    print("\nPASS: every child script in the harness binds every name it uses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
