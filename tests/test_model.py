# ============================================================
# 11.2 — MODEL CLASS COUNT VALIDATION
# ============================================================

from src.inference.predict import load_class_names


def test_model_has_expected_classes():

    class_names = load_class_names()

    assert len(class_names) == 10


def test_class_names_are_strings():

    class_names = load_class_names()

    assert all(
        isinstance(name, str)
        for name in class_names
    )


def test_class_names_are_not_empty():

    class_names = load_class_names()

    assert all(
        name.strip()
        for name in class_names
    )