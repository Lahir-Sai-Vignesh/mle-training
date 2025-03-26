import importlib


def test_import_evaluation():
    assert (
        importlib.util.find_spec("my_package.evaluation") is not None
    ), "Module 'evaluation' not found"
