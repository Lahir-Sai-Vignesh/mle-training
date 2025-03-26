import importlib


def test_import_training():
    assert (
        importlib.util.find_spec("my_package.training") is not None
    ), "Module 'training' not found"
