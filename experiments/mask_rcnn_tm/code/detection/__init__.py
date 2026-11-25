"""Detection machinery shared by the Tsetlin pipeline and baseline (iv): pure tensor code, no
Tsetlin imports (PLAN_KVASIR §5; module tests in ``tests/test_detection.py``).

Boxes are ``x1, y1, x2, y2`` in pixel-edge coordinates: pixel ``i`` covers ``[i, i + 1)``, so the
tight box of a mask is ``(min, min, max + 1, max + 1)``.
"""
