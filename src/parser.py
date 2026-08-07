from typing import Any

import numpy as np
import sympy as sp


class InvalidExpressionError(Exception):
    """Исключение при синтаксической ошибке в формуле"""


class CompiledExecutable:
    """
    Отвечает за вызов numpy-инструкции
    По сути берет скомпилированную функцию, вектор иксов
    выдает игреки
    """

    def __init__(self, raw_func: Any):
        self.raw_func = raw_func

    def __call__(self, x_arr: np.ndarray) -> np.ndarray:
        try:
            with np.errstate(all="ignore"):
                raw_res = self.raw_func(x_arr)

            # на случай если ф-ия вернула константу
            if isinstance(raw_res, (int, float, np.number)):
                raw_res = np.full_like(x_arr, fill_value=float(raw_res))

            res = np.asarray(raw_res, dtype=np.float64)

            # заменяем неопределенности на чиселки
            return np.nan_to_num(res, nan=0.0, posinf=1e8, neginf=-1e8)

        except Exception as e:
            raise InvalidExpressionError(f"Мат. ошибка при расчете f(x): {e}") from e


class ExpressionParser:
    """Парсер мат. выражений"""

    MATH_FUNCTIONS = (
        # знаки, корни
        "abs",
        "sqrt",
        "cbrt",
        "sign",
        # трига
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
        "asin",
        "acos",
        "atan",
        "acot",
        # гиперболическая трига
        "sinh",
        "cosh",
        "tanh",
        "coth",
        "asinh",
        "acosh",
        "atanh",
        # экспоненты и логарифмы
        "exp",
        "log",
        "ln",
        # ступенчатые ф-ии
        "floor",
        "ceiling",
        # кака-ято фигня красивая
        "erf",
        "erfc",
        "gamma",
        "sinc",
    )

    MATH_CONSTANTS = {  # noqa: RUF012
        "pi": sp.pi,
        "e": sp.E,
        "tau": 2 * sp.pi,
    }

    @classmethod
    def _get_allowed_math(cls) -> dict[str, Any]:
        allowed = {}

        for fname in cls.MATH_FUNCTIONS:
            if hasattr(sp, fname):
                allowed[fname] = getattr(sp, fname)

        allowed["ln"] = sp.log
        allowed["log2"] = lambda x: sp.log(x, 2)
        allowed["log10"] = lambda x: sp.log(x, 10)

        allowed.update(cls.MATH_CONSTANTS)

        return allowed

    @classmethod
    def parse(cls, expr_str: str):
        """
        Превращает текстовую ф-лу в скомпилированный объект для вычислений
        """

        if not expr_str or not expr_str.strip():
            raise InvalidExpressionError("Пустое выражение передали :(")

        cleaned_expr_str = expr_str.strip()
        x = sp.Symbol("x")
        allowed_math = cls._get_allowed_math()

        # переводим в выражение
        try:
            expr = sp.sympify(
                cleaned_expr_str,
                locals={"x": x, **allowed_math},
                strict=True,
            )
        except Exception as e:
            raise InvalidExpressionError(
                f"Синтаксическая ошибка в формуле '{cleaned_expr_str}': {e}"
            ) from e

        # чекаем на другие переменные
        free_symbols = expr.free_symbols
        invalid_symbols = free_symbols - {x}

        if invalid_symbols:
            invalid_names = sorted([s.name for s in invalid_symbols])
            raise InvalidExpressionError(
                f"В выражении есть разные переменные: {invalid_names}. "
                f"Пока разрешен только 'x'"
            )

        # компилируем
        try:
            raw_func = sp.lambdify(x, expr, modules=["scipy", "numpy"])
        except Exception as e:
            raise InvalidExpressionError(f"Ошибка при компиляции ф-ии :( {e}") from e

        return CompiledExecutable(raw_func)
