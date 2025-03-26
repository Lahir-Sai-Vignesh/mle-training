import importlib


def test_import_data_ingestion():
    assert (
        importlib.util.find_spec("my_package.data_ingestion") is not None
    ), "Module 'data_ingestion' not found"
