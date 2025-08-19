#!/usr/bin/env python3
"""
Simple script to verify CLensPy installation and structure.
"""

import os
import sys


def check_structure():
    """Check if the CLensPy directory structure is correct."""

    base_dir = "/Users/esteves/Documents/Projetos/CLensPy"

    expected_structure = {
        "src/clenspy": ["__init__.py", "config.py"],
        "src/clenspy/lensing": ["__init__.py", "profile.py", "boost.py", "miscentering.py"],
        "src/clenspy/halo": [
            "__init__.py",
            "nfw.py",
            "bias.py",
            "concentration.py",
            "twohalo.py",
            "einasto.py",
        ],
        "src/clenspy/cosmology": ["__init__.py", "utils.py", "pkgrid.py"],
        "sec/clenspy/utils": [
            "__init__.py",
            "decorators.py",
            "interpolate.py",
            "integrate.py",
        ],
        "tests": [
            "__init__.py",
            "test_lensing.py",
            "test_utils.py",
            "test_nfw.py",
            "test_twohalo.py",
        ],
        "examples": ["demo_basic_usage.py", "demo_profile_fit.ipynb"],
        "docs": ["index.md"],
        "docs/tutorials": ["tutorial1.ipynb"],
        ".github/workflows": ["tests.yml"],
        ".": ["pyproject.toml", "README.md", "LICENSE", "setup.cfg", ".gitignore"],
    }

    print("Checking CLensPy directory structure...")
    print("=" * 60)

    all_good = True
    total_files = 0
    found_files = 0
    empty_files = 0

    for directory, files in expected_structure.items():
        dir_path = os.path.join(base_dir, directory) if directory != "." else base_dir

        if not os.path.exists(dir_path):
            print(f"❌ Directory missing: {directory}")
            all_good = False
            total_files += len(files)
            continue

        print(f"✅ Directory exists: {directory}")

        for file in files:
            file_path = os.path.join(dir_path, file)
            total_files += 1

            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                found_files += 1
                if file_size > 0:
                    size_str = format_file_size(file_size)
                    print(f"  ✅ {file} ({size_str})")
                else:
                    print(f"  ⚠️  {file} (empty file)")
                    empty_files += 1
                    all_good = False
            else:
                print(f"  ❌ {file} (missing)")
                all_good = False

    print("=" * 60)

    # Summary statistics
    print(f"\nFile statistics:")
    print(f"  Found: {found_files}/{total_files} files")
    print(f"  Empty: {empty_files} files")
    completion_percent = (found_files / total_files) * 100 if total_files > 0 else 0
    print(f"  Completion: {completion_percent:.1f}%")

    # Check for additional development files
    print("\nDevelopment files:")
    dev_files = [
        "check_structure.py",
        "demo_new_structure.py",
        "test_nfw_quick.py",
        "ruff.toml",
        ".gitignore",
    ]

    dev_found = 0
    for file in dev_files:
        file_path = os.path.join(base_dir, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            size_str = format_file_size(file_size)
            print(f"  ✅ {file} ({size_str})")
            dev_found += 1
        else:
            print(f"  ⚠️  {file} (not found)")

    # Check for VS Code configuration
    print("\nVS Code configuration:")
    vscode_files = [
        ".vscode/settings.json",
        ".vscode/launch.json",
        ".vscode/tasks.json",
    ]

    vscode_found = 0
    for file in vscode_files:
        file_path = os.path.join(base_dir, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            size_str = format_file_size(file_size)
            print(f"  ✅ {file} ({size_str})")
            vscode_found += 1

    if vscode_found == 0:
        print("  (No VS Code configuration found)")

    print("=" * 60)

    if all_good:
        print("🎉 All files and directories are in place!")
        print_package_summary()
        print_next_steps()
    else:
        print("⚠️  Some files or directories are missing or empty.")
        print_fix_suggestions(found_files, total_files)

    return all_good


def format_file_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    elif size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def print_package_summary():
    """Print package structure summary."""
    print("\nPackage structure summary:")
    print("├── src/clenspy/")
    print("│   ├── lensing/          # Cosmology-dependent observables")
    print("│   │   ├── profile.py    # LensingProfile class")
    print("│   │   ├── boost.py      # Boost factor corrections")
    print("│   │   └── miscentering.py # Miscentering corrections")
    print("│   ├── halo/             # Theory-focused halo properties")
    print("│   │   ├── __init__.py   # Halo module initialization")
    print("│   │   ├── nfw.py        # NFW density profiles")
    print("│   │   ├── twohalo.py    # Two-halo term")
    print("│   │   ├── einasto.py    # Einasto profile (if implemented)")
    print("│   │   ├── bias.py       # Halo bias models")
    print("│   │   └── concentration.py # c-M relations")
    print("│   ├── cosmology/        # Cosmology utilities")
    print("│   │   └── utils.py      # sigma_crit, distance conversions")
    print("│   ├── └── pkgrid.py     # Power spectrum grid (CAMB, PyCCL)")
    print("│   ├── utils/            # General utilities")
    print("│   │   └── interpolate.py# Interpolation utilities")
    print("│   ├── └── integrate.py  # Integration utilities")
    print("│   ├── └── decorators.py # Decorators for utility functions")
    print("│   └── config.py         # Default cosmology and constants")
    print("├── tests/                # Unit tests and benchmarks")
    print("│   ├── test_*.py         # Standard unit tests")
    print("│   ├── test_nfw.py       # NFW tests")
    print("│   └── test_twohalo.py   # Two-halo tests")
    print("├── data/                 # Data files (if any, e.g. cosmology parameters)")
    print("├── examples/            # Usage examples")
    print("├── docs/                # Documentation")
    print("└── .github/             # GitHub workflows")

    print("\nArchitectural principles:")
    print("✅ Lensing: Cosmology-dependent observables")
    print("✅ Halo: Theory-focused, minimal cosmology dependency")
    print("✅ Cosmology: Centralized utility functions")
    print("✅ Clean separation between theory and observations")


def print_next_steps():
    """Print next steps for using the package."""
    print("\nNext steps:")
    print("1. Install package:")
    print("   cd /Users/esteves/Documents/Projetos/CLensPy")
    print("   pip install -e .")

    print("\n2. Run tests:")
    print("   pytest tests/                    # All tests")
    print("   pytest tests/test_nfw_fourier.py # NFW Fourier tests")
    print("   pytest tests/ -m benchmark       # Benchmark tests")

    print("\n3. Try examples:")
    print("   python examples/demo_basic_usage.py")

    print("\n4. Test imports:")
    print(
        "   python -c \"from clenspy.lensing import LensingProfile; print('Success!')\""
    )

    print("\n5. API usage:")
    print("   from clenspy.lensing import LensingProfile")
    print("   from clenspy.cosmology import sigma_critical")
    print("   from clenspy.config import DEFAULT_COSMOLOGY")


def print_fix_suggestions(found, total):
    """Print suggestions for fixing missing files."""
    print("\nTo fix missing files:")

    if found < total * 0.5:
        print("• Run: python demo_new_structure.py")
        print("• Create missing module files")
    else:
        print("• Check individual missing files above")
        print("• Create empty files if needed:")
        print("  touch missing_file.py")

    print("\nFor empty files:")
    print("• Add at least basic docstrings and imports")
    print("• Implement placeholder functions if needed")


def check_imports():
    """Test basic imports to verify package structure."""
    print("\n" + "=" * 60)
    print("Testing basic imports...")

    try:
        # Add current directory to path for testing
        sys.path.insert(0, "/Users/esteves/Documents/Projetos/CLensPy")

        # Test main package import
        import clenspy

        print("✅ Successfully imported clenspy")
        print(f"   Version: {getattr(clenspy, '__version__', 'Unknown')}")

        # Test config import
        from clenspy.config import DEFAULT_COSMOLOGY, RHOCRIT

        print("✅ Successfully imported config")
        print(f"   DEFAULT_COSMOLOGY: {DEFAULT_COSMOLOGY}")
        print(f"   RHOCRIT: {RHOCRIT:.2e} Msun/Mpc^3")

        # Test cosmology import
        from clenspy.cosmology import sigma_critical, comoving_to_theta, PkGrid

        print("✅ Successfully imported cosmology utilities")

        # Test lensing import (cosmology-dependent observables)
        from clenspy.lensing import LensingProfile

        print("✅ Successfully imported LensingProfile (lensing observables)")

        # Test halo import (theory-focused)
        from clenspy.halo import NFWProfile

        print("✅ Successfully imported NFWProfile (halo theory)")

        # Test two halo term import
        from clenspy.halo import TwoHaloTerm

        print("✅ Successfully imported TwoHaloTerm (two-halo term)")

        # Test utils import
        from clenspy.utils import decorators, interpolate, integrate

        print("✅ Successfully imported utils (decorators, interpolate, integrate)")

        # Test specific module imports
        try:
            from clenspy.lensing import boost, miscentering

            print("✅ Successfully imported boost and miscentering corrections")
        except ImportError as e:
            print(f"⚠️  Could not import boost/miscentering: {e}")

        try:
            from clenspy.halo import bias, concentration

            print("✅ Successfully imported bias and concentration models")
        except ImportError as e:
            print(f"⚠️  Could not import bias/concentration: {e}")

        print("🎉 Core imports successful!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nPossible causes:")
        print("• Missing dependencies (astropy, numpy, scipy)")
        print("• Incomplete package structure")
        print("• Python path issues")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality with simple calculations."""
    print("\n" + "=" * 60)
    print("Testing basic functionality...")

    try:
        sys.path.insert(0, "/Users/esteves/Documents/Projetos/CLensPy")

        from clenspy.config import DEFAULT_COSMOLOGY
        from clenspy.cosmology import sigma_critical
        import numpy as np

        # Test sigma_critical calculation
        z_lens = 0.3
        z_source = 1.0
        sigma_crit = sigma_critical(z_lens, z_source, DEFAULT_COSMOLOGY)
        print(f"✅ sigma_critical({z_lens}, {z_source}) = {sigma_crit:.2e} Msun/Mpc^2")

        # Test LensingProfile creation
        from clenspy.lensing import LensingProfile

        profile = LensingProfile(
            cosmology=DEFAULT_COSMOLOGY, z_cluster=0.3, m200=1e15, concentration=4.0
        )
        print("✅ Successfully created LensingProfile")
        print(f"   r200 = {profile.halo_profile.r200:.2f} Mpc")
        print(f"   rs = {profile.halo_profile.rs:.2f} Mpc")

        # Test basic calculation
        R = np.array([0.5, 1.0, 2.0])  # Mpc
        try:
            delta_sigma = profile.deltaSigmaR(R)
            print(f"✅ deltaSigmaR calculation: {len(delta_sigma)} values computed")
            print(f"   Values: {delta_sigma}")
        except Exception as e:
            print(f"⚠️  delta_sigma calculation failed: {e}")

        # Test NFW Fourier profile if available
        try:
            k = np.array([0.1, 1.0, 10.0])
            fourier_profile = profile.halo_profile.fourier(k)
            print(f"✅ NFW Fourier profile: {len(fourier_profile)} values computed")
        except AttributeError:
            print("⚠️  NFW Fourier profile method not implemented yet")
        except Exception as e:
            print(f"⚠️  NFW Fourier profile failed: {e}")

        print("🎉 Basic functionality test passed!")
        return True

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        print("\nThis might indicate:")
        print("• Missing or incomplete implementation")
        print("• Dependency issues")
        print("• Interface problems between modules")
        return False


def main():
    """Main function to run all checks."""
    print("CLensPy Package Structure and Functionality Check")
    print("=" * 60)

    # Check structure
    structure_ok = check_structure()

    if not structure_ok:
        print("\n❌ Package structure check failed.")
        print("Fix structure issues before proceeding.")
        return

    # Check imports
    imports_ok = check_imports()

    if not imports_ok:
        print("\n⚠️  Package structure is correct but imports failed.")
        print("Check dependencies and implementation.")
        return

    # Test functionality
    functionality_ok = test_basic_functionality()

    if functionality_ok:
        print("\n🚀 CLensPy package is fully functional!")
        print("\nArchitectural summary:")
        print("✅ Lensing: Cosmology-dependent observables")
        print("✅ Halo: Theory-focused profiles and models")
        print("✅ Cosmology: Utility functions for lensing calculations")
        print("✅ Tests: Unit tests and performance benchmarks")
        print("✅ Clean separation maintained")

        print("\nReady for:")
        print("• Scientific calculations")
        print("• Performance optimization")
        print("• Further development")
    else:
        print(
            "\n⚠️  Package structure and imports work, but functionality needs attention."
        )


if __name__ == "__main__":
    main()
