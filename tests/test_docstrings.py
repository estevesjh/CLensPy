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


def silent_classes(src):
    """Names of API classes in ``src`` with no unit convention declared.

    Split out from the test so `test_the_units_check_has_teeth` can drive
    it directly on synthetic sources -- an exemption that silences the
    whole check would otherwise pass unnoticed.
    """
    tree = ast.parse(src)

    def declares(doc):
        return bool(doc) and "NOTE:" in doc and any(w in doc for w in UNIT_WORDS)

    def is_main_guard(node):
        return (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        )

    # classes defined under `if __name__ == "__main__":` are demo fixtures
    demo_only = {
        cls.name
        for node in ast.walk(tree)
        if is_main_guard(node)
        for cls in ast.walk(node)
        if isinstance(cls, ast.ClassDef)
    }

    if declares(ast.get_docstring(tree)):
        return []
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name not in demo_only
        and not declares(ast.get_docstring(node))
    ]


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_every_class_declares_its_units(path):
    """Each class states its unit convention, or its module does for it.

    The `cosmology-code` rule is "declare units at the class". A class in a
    module whose own docstring carries the convention satisfies it too --
    that is how `clenspy.selection.miscentering` and `clenspy.utils.binning`
    do it, and it keeps the declaration in one place per module rather than
    repeated per class.

    NOTE: classes defined inside a module's ``if __name__ == "__main__"``
    block are exempt. The rule is about the package's API surface -- what a
    caller holds and passes physical quantities to -- and a demo fixture is
    neither importable nor given units. `clenspy.utils.decorators` is the
    case: a pure-mechanism module with no physical units anywhere, whose
    demo needs some class to hang the decorators on.
    """
    silent = silent_classes(path.read_text())
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


def test_the_units_check_has_teeth():
    """The check catches a silent API class, and the module escape works."""
    assert silent_classes('class Silent:\n    """Nothing."""\n') == ["Silent"]
    # a class that declares for itself
    assert silent_classes(
        'class Loud:\n    """NOTE: all masses in Msun."""\n'
    ) == []
    # a module that declares for its classes
    assert silent_classes(
        '"""NOTE: units are Mpc throughout."""\n'
        'class Quiet:\n    """Nothing."""\n'
    ) == []


def test_the_demo_exemption_is_scoped_to_the_main_guard():
    """The exemption must not silence the check for real classes.

    Same class body, two placements: module level is caught, nested under
    the ``__main__`` guard is not. Without this, adding the exemption could
    have disabled the rule package-wide and every test would still pass.
    """
    body = 'class Demo:\n    """Nothing."""\n'
    assert silent_classes(body) == ["Demo"]
    assert silent_classes(
        'if __name__ == "__main__":\n    ' + body.replace("\n    ", "\n        ")
    ) == []
    # and a real class is still caught when a demo fixture sits below it
    assert silent_classes(
        body + 'if __name__ == "__main__":\n    class Fixture:\n'
        '        """Nothing."""\n'
    ) == ["Demo"]
