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
        "clenspy": ["__init__.py", "config.py"],
        "clenspy/lensing": ["__init__.py", "profile.py", "boost.py", "miscentering.py"],
        "clenspy/halo": ["__init__.py", "nfw.py", "bias.py", "concentration.py"],
        "clenspy/cosmology": ["__init__.py", "utils.py"],
        "clenspy/utils": ["__init__.py", "coordinates.py"],
        "tests": [
            "__init__.py",
            "test_lensing.py",
            "test_halo.py",
            "test_cosmology.py",
            "test_utils.py",
        ],
        "examples": ["demo_basic_usage.py", "demo_profile_fit.ipynb"],
        "docs": ["index.md"],
        "docs/tutorials": ["tutorial1.ipynb"],
        ".github/workflows": ["tests.yml"],
        ".": ["pyproject.toml", "README.md", "LICENSE", "setup.cfg", ".gitignore"],
    }

    print("Checking CLensPy directory structure...")
    print("=" * 50)

    all_good = True

    for directory, files in expected_structure.items():
        dir_path = os.path.join(base_dir, directory) if directory != "." else base_dir

        if not os.path.exists(dir_path):
            print(f"❌ Directory missing: {directory}")
            all_good = False
            continue

        print(f"✅ Directory exists: {directory}")

        for file in files:
            file_path = os.path.join(dir_path, file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > 0:
                    print(f"  ✅ {file} ({file_size} bytes)")
                else:
                    print(f"  ⚠️  {file} (empty file)")
            else:
                print(f"  ❌ {file} (missing)")
                all_good = False

    print("=" * 50)
    
    # Check for additional files that might exist
    print("\nAdditional files found:")
    additional_files = [
        "check_structure.py",
        "demo_new_structure.py",
        ".gitignore",
        "ruff.toml",
    ]
    
    for file in additional_files:
        file_path = os.path.join(base_dir, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  ✅ {file} ({file_size} bytes)")

    print("=" * 50)
    
    if all_good:
        print("🎉 All files and directories are in place!")
        print("\nPackage structure summary:")
        print("├── clenspy/")
        print("│   ├── lensing/          # LensingProfile class and corrections")
        print("│   ├── halo/             # NFW profiles, bias, concentration")
        print("│   ├── cosmology/        # Cosmology utilities (sigma_crit, etc.)")
        print("│   ├── utils/            # Coordinate conversions, utilities")
        print("│   └── config.py         # Default cosmology and constants")
        print("├── tests/               # Unit tests")
        print("├── examples/            # Usage examples")
        print("├── docs/                # Documentation")
        print("└── .github/             # GitHub workflows")
        
        print("\nNext steps:")
        print("1. cd /Users/esteves/Documents/Projetos/CLensPy")
        print("2. pip install -e .")
        print("3. pytest tests/")
        print("4. python examples/demo_basic_usage.py")
        print("5. Try the new API:")
        print("   from clenspy.lensing import LensingProfile")
        print("   from clenspy.config import DEFAULT_COSMOLOGY")
        print("   from clenspy.cosmology import sigma_critical")
    else:
        print("⚠️  Some files or directories are missing.")
        print("\nTo create missing files, run:")
        print("python demo_new_structure.py")

    return all_good


def check_imports():
    """Test basic imports to verify package structure."""
    print("\n" + "=" * 50)
    print("Testing basic imports...")
    
    try:
        # Add current directory to path for testing
        sys.path.insert(0, "/Users/esteves/Documents/Projetos/CLensPy")
        
        # Test main package import
        import clenspy
        print("✅ Successfully imported clenspy")
        
        # Test config import
        from clenspy.config import DEFAULT_COSMOLOGY, RHOCRIT
        print("✅ Successfully imported config")
        print(f"   DEFAULT_COSMOLOGY: {DEFAULT_COSMOLOGY}")
        print(f"   RHOCRIT: {RHOCRIT:.2e}")
        
        # Test cosmology import
        from clenspy.cosmology import sigma_critical, comoving_to_theta
        print("✅ Successfully imported cosmology utilities")
        
        # Test lensing import
        from clenspy.lensing import LensingProfile
        print("✅ Successfully imported LensingProfile")
        
        # Test halo import
        from clenspy.halo import NFWProfile
        print("✅ Successfully imported NFWProfile")
        
        # Test utils import
        from clenspy.utils import coordinates
        print("✅ Successfully imported utils.coordinates")
        
        print("🎉 All imports successful!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    structure_ok = check_structure()
    
    if structure_ok:
        imports_ok = check_imports()
        if imports_ok:
            print("\n🚀 CLensPy package is ready to use!")
        else:
            print("\n⚠️  Package structure is correct but imports failed.")
            print("This might be due to missing dependencies (astropy, numpy, scipy)")
    else:
        print("\n❌ Package structure needs to be fixed first.")
