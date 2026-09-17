# Adapted from https://stackoverflow.com/a/9558001/2536294

import ast
import functools
import operator as op
from dataclasses import dataclass

from ._multiprocessing_helpers import mp

if mp is not None:
    from .externals.loky.process_executor import _ExceptionWithTraceback


# supported operators
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

# Largest magnitude any sub-expression is allowed to reach.
MAX_VALUE = 10**6


def _pow_is_too_large(base, exponent):
    """Return True if ``base ** exponent`` cannot fit under `MAX_VALUE`.

    Only integers are considered. A float power overflows cheaply, whereas a
    large integer power is built out in full before `limit` ever sees the
    result: ``999999**999999`` costs seconds of CPU and allocates a number with
    millions of digits.
    """
    if not (isinstance(base, int) and isinstance(exponent, int)):
        return False
    # For any |base| >= 2, base ** exponent >= 2 ** exponent, so an exponent
    # above the bit length of the limit cannot produce a value that fits.
    return abs(base) >= 2 and exponent > MAX_VALUE.bit_length()


def eval_expr(expr):
    """Somewhat safely evaluate an arithmetic expression.

    >>> eval_expr('2*6')
    12
    >>> eval_expr('2**6')
    64
    >>> eval_expr('1 + 2*3**(4) / (6 + -7)')
    -161.0

    Raises ValueError if the expression is invalid, too long
    or its computation involves too large values.
    """
    # Restrict the length of the expression to avoid potential Python crashes
    # as per the documentation of ast.parse.
    max_length = 30
    if len(expr) > max_length:
        raise ValueError(
            f"Expression {expr[:max_length]!r}... is too long. "
            f"Max length is {max_length}, got {len(expr)}."
        )
    try:
        return eval_(ast.parse(expr, mode="eval").body)
    except (TypeError, SyntaxError, OverflowError, KeyError, ZeroDivisionError) as e:
        raise ValueError(
            f"{expr!r} is not a valid or supported arithmetic expression."
        ) from e


def limit(max_=None):
    """Return decorator that limits allowed returned values."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            try:
                mag = abs(ret)
            except TypeError:
                pass  # not applicable
            else:
                if mag > max_:
                    raise ValueError(
                        f"Numeric literal {ret} is too large, max is {max_}."
                    )
            return ret

        return wrapper

    return decorator


@limit(max_=MAX_VALUE)
def eval_(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        left = eval_(node.left)
        right = eval_(node.right)
        if isinstance(node.op, ast.Pow) and _pow_is_too_large(left, right):
            raise ValueError(
                f"Numeric literal {left}**{right} is too large, max is {MAX_VALUE}."
            )
        return operators[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return operators[type(node.op)](eval_(node.operand))
    else:
        raise TypeError(node)


@dataclass(frozen=True)
class _Sentinel:
    """A sentinel to mark a parameter as not explicitly set"""

    default_value: object

    def __repr__(self):
        return f"default({self.default_value!r})"


class _TracebackCapturingWrapper:
    """Protect function call and return error with traceback."""

    def __init__(self, func):
        self.func = func

    def __call__(self, **kwargs):
        try:
            return self.func(**kwargs)
        except BaseException as e:
            return _ExceptionWithTraceback(e)


def _retrieve_traceback_capturing_wrapped_call(out):
    if isinstance(out, _ExceptionWithTraceback):
        rebuild, args = out.__reduce__()
        out = rebuild(*args)
    if isinstance(out, BaseException):
        raise out
    return out
