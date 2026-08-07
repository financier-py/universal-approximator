from typing import Any

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

# ========== имена для функций тут будут ==========
_MATH_FUNCTIONS = (
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

# принятые обозначения в РФ
_RU_ALIASES = {
    # тангенс катангенс
    "tg": sp.tan,
    "ctg": sp.cot,
    # арк трига
    "arctg": sp.atan,
    "arcctg": sp.acot,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    # гиперболические
    "sh": sp.sinh,
    "ch": sp.cosh,
    "th": sp.tanh,
    "cth": sp.coth,
    # арк гиперболические
    "arsh": sp.asinh,
    "arch": sp.acosh,
    "arth": sp.atanh,
    "arcth": sp.acoth,
    # Логарифм по умолчанию
    "ln": sp.log,
}

_MATH_CONSTANTS = {
    "pi": sp.pi,
    "e": sp.E,
    "tau": 2 * sp.pi,
}

_X_SYM = sp.Symbol("x")
_LOG_ALIASES = {
    "lg": sp.log(_X_SYM, 10),
    "log10": sp.log(_X_SYM, 10),
    "log2": sp.log(_X_SYM, 2),
}

# итоговый словарь разрешенных мат. ф-ий
_ALLOWED_MATH = (
    {fname: getattr(sp, fname) for fname in _MATH_FUNCTIONS if hasattr(sp, fname)}
    | _RU_ALIASES
    | _LOG_ALIASES
    | _MATH_CONSTANTS
)


class InvalidExpressionError(Exception):
    """Исключение при синтаксической ошибке в формуле"""


class FunctionEvaluator:
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

    @classmethod
    def parse(cls, expr_str: str):
        """
        Превращает текстовую ф-лу в скомпилированный объект для вычислений
        """

        if not expr_str or not expr_str.strip():
            raise InvalidExpressionError("Пустое выражение передали :(")

        cleaned_expr_str = expr_str.strip()
        x = sp.Symbol("x")

        # переводим в выражение
        try:
            transformations = standard_transformations + (convert_xor,)
            expr = parse_expr(
                cleaned_expr_str,
                local_dict={"x": x, **_ALLOWED_MATH},
                transformations=transformations,
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
            raw_func = sp.lambdify(x, expr, modules=["numpy"])
        except Exception as e:
            raise InvalidExpressionError(f"Ошибка при компиляции ф-ии :( {e}") from e

        return FunctionEvaluator(raw_func)
