# ============================================================
# 11.1 — BASIC PREDICTION TEST
# ============================================================

from src.inference.predict import load_class_names


def test_class_names():

    class_names = load_class_names()

    assert class_names is not None
    assert len(class_names) == 10