import pytest

from joblib._utils import eval_expr


@pytest.mark.parametrize(
    "expr, error, message",
    [
        ("exec('import os')", TypeError, "ast.Call object"),
        ("print(1)", TypeError, "ast.Call object"),
        ("import os", SyntaxError, "invalid syntax"),
        ("1+1; import os", SyntaxError, "invalid syntax"),
        ("1^1", KeyError, "class 'ast.BitXor'"),
    ],
)
def test_eval_expr_invalid(expr, error, message):
    with pytest.raises(error, match=message):
        eval_expr(expr)


@pytest.mark.parametrize(
    "expr, result", [("2*6", 12), ("2**6", 64), ("1 + 2*3**(4) / (6 + -7)", -161.0)]
)
def test_eval_expr_valid(expr, result):
    assert eval_expr(expr) == result
