import importlib


def test_package_import():
    """Verify the package is correctly installed by attempting to import it."""
    package_name = "my_package"
    try:
        module = importlib.import_module(package_name)
        assert module is not None
    except ModuleNotFoundError:
        assert False, f"Failed to import {package_name}"
