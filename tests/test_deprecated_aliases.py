"""Tests for the module-level ``__getattr__`` backward-compatibility shims.

Two modules have a lazy deprecated alias:

- ``clenspy.lensing.boost`` -> ``clenspy.selection.boost``
- ``clenspy.cosmology.sigma_critical`` -> ``clenspy.kernels.sigma_crit.sigma_critical``

Both should emit a ``DeprecationWarning`` and resolve to the new location,
while any other unknown attribute should still raise ``AttributeError``.
"""

import pytest

import clenspy.cosmology as cosmology
import clenspy.lensing as lensing


def test_lensing_boost_alias_warns_and_resolves():
    with pytest.warns(DeprecationWarning, match="clenspy.lensing.boost moved to"):
        mod = lensing.boost

    import clenspy.selection.boost as boost_mod

    assert mod is boost_mod
    assert mod.__name__ == "clenspy.selection.boost"


def test_lensing_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        lensing.nonexistent_attr


def test_cosmology_sigma_critical_alias_warns_and_resolves():
    from clenspy.kernels.sigma_crit import sigma_critical as sc

    with pytest.warns(
        DeprecationWarning, match="clenspy.cosmology.sigma_critical moved to"
    ):
        fn = cosmology.sigma_critical

    assert fn is sc


def test_cosmology_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        cosmology.nonexistent_attr
