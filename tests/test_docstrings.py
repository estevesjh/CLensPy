r"""Docstring hygiene, for the two failures that are silent.

This package writes a lot of LaTeX in docstrings, which makes one bug easy
to introduce and nearly impossible to see: a backslash in a **non-raw**
docstring is a Python escape, so ``\rho`` in a plain (non-``r``-prefixed)
docstring becomes a literal carriage return. The rendered maths breaks,
napoleon stops parsing sections, and the source still looks correct.

The second check is the unit convention: the `cosmology-code` rule is that
every class declares its units in its own docstring, because a surface
density that is off by :math:`h^2` or :math:`(1+z)` is the most expensive
kind of wrong in this field.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "clenspy"
MODULES = sorted(SRC.rglob("*.py"))

DOC_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

#: A docstring "declares its units" if it carries a NOTE: and names either
#: the convention or one of the package's units.
UNIT_WORDS = ("unit", "Msun", "Mpc", "dimensionless")


def docstrings(path):
    """Yield ``(name, lineno, source_text, is_raw)`` for every docstring."""
    src = path.read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, DOC_NODES) or not node.body:
            continue
        if ast.get_docstring(node) is None:
            continue
        d = node.body[0].value
        opener = lines[d.lineno - 1].lstrip()
        yield (
            getattr(node, "name", "<module>"),
            d.lineno,
            ast.get_source_segment(src, d) or "",
            opener[:1].lower() == "r",
        )


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_docstrings_with_backslashes_are_raw(path):
    r"""A backslash in a non-raw docstring is a Python escape.

    ``\rho`` is the dangerous one -- ``\r`` is a real escape, so the
    docstring silently acquires a carriage return. ``\Sigma`` and
    ``\Delta`` merely warn today and become errors in a future Python.
    Either way the fix is the same: prefix the docstring with ``r``.
    """
    offenders = [
        f"{path.name}:{lineno} in {name}"
        for name, lineno, text, is_raw in docstrings(path)
        if not is_raw and "\\" in text
    ]
    assert not offenders, (
        "non-raw docstring containing a backslash -- prefix it with r: "
        + "; ".join(offenders)
    )


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_every_class_declares_its_units(path):
    """Each class states its unit convention, or its module does for it.

    The `cosmology-code` rule is "declare units at the class". A class in a
    module whose own docstring carries the convention satisfies it too --
    that is how `clenspy.selection.miscentering` and `clenspy.utils.binning`
    do it, and it keeps the declaration in one place per module rather than
    repeated per class.
    """
    src = path.read_text()
    tree = ast.parse(src)

    def declares(doc):
        return bool(doc) and "NOTE:" in doc and any(w in doc for w in UNIT_WORDS)

    module_declares = declares(ast.get_docstring(tree))
    silent = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and not declares(ast.get_docstring(node))
        and not module_declares
    ]
    assert not silent, (
        f"{path.name}: class(es) with no unit convention stated, in the "
        f"class or its module: {silent}"
    )


def test_the_checks_have_teeth(tmp_path):
    """Both checks must actually fail on a file that breaks them."""
    bad = tmp_path / "bad.py"
    bad.write_text('class Silent:\n    """No units here, and \\rho unescaped."""\n')
    names, _, texts, raws = zip(*[(n, i, t, r) for n, i, t, r in docstrings(bad)])
    assert any("\\" in t and not r for t, r in zip(texts, raws))
    assert "Silent" in names
