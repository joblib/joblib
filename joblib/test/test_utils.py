import time

import pytest

from joblib._utils import eval_expr


@pytest.mark.parametrize(
    "expr",
    [
        "exec('import os')",
        "print(1)",
        "import os",
        "1+1; import os",
        "1^1",
        "' ' * 10**10",
        "9. ** 10000.",
        "1/0",
        "1//0",
        "1%0",
    ],
)
def test_eval_expr_invalid(expr):
    with pytest.raises(ValueError, match="is not a valid or supported arithmetic"):
        eval_expr(expr)


def test_eval_expr_too_long():
    expr = "1" + "+1" * 50
    with pytest.raises(ValueError, match="is too long"):
        eval_expr(expr)


@pytest.mark.parametrize(
    "expr",
    [
        "1e7",
        "10**7",
        "9**9**9",
        # A result with more digits than sys.get_int_max_str_digits() allows cannot be
        # rendered, so reporting it has to not depend on formatting the value.
        "10**5000",
        # Both operands are within the limit, but the power itself is not.
        # Evaluating it takes seconds and allocates millions of digits, so it
        # has to be rejected before it is computed.
        "999999**999999",
    ],
)
def test_eval_expr_too_large_literal(expr):
    with pytest.raises(ValueError, match="Numeric literal .* is too large"):
        eval_expr(expr)


def test_eval_expr_oversized_power_is_not_evaluated():
    # Guards against a regression to computing the value before checking it.
    # Evaluating this expression takes seconds and allocates a number with
    # millions of digits, so a slow run here is the failure being tested for.
    start = time.perf_counter()
    with pytest.raises(ValueError, match="Numeric literal .* is too large"):
        eval_expr("999999**999999")
    assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize(
    "expr, result",
    [
        ("2*6", 12),
        ("2**6", 64),
        ("1 + 2*3**(4) / (6 + -7)", -161.0),
        ("(20 // 3) % 5", 1),
        # Powers that do fit are still evaluated, including at the limit itself.
        ("2**19", 524288),
        ("10**6", 1000000),
        # A base of magnitude one stays small whatever the exponent.
        ("1**999999", 1),
        ("999999**0", 1),
    ],
)
def test_eval_expr_valid(expr, result):
    assert eval_expr(expr) == result
