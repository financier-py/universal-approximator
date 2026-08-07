from dataclasses import dataclass

import numpy as np
import torch

from src.parser import FunctionEvaluator
from src.schemas.experiment_params import ExperimentParams


@dataclass(frozen=True)
class PreparedData:
    """Все необходимые данные для одного эксперимента"""

    # для обучения PyTorch (N на 1)
    x_train_tensor: torch.Tensor
    y_train_tensor: torch.Tensor

    # валидация модели
    has_val_points: bool
    x_val_tensor: torch.Tensor | None
    y_val_tensor: torch.Tensor | None

    # для визуализации
    x_all_tensor: torch.Tensor

    x_grid: np.ndarray
    y_target: np.ndarray
    train_mask: np.ndarray


class DataBuilder:
    """Просто генератор сетки точек по нашей f(x)"""

    @staticmethod
    def create(params: ExperimentParams, func: FunctionEvaluator) -> PreparedData:
        # генерим точки всей видимой области
        x_grid = np.linspace(
            params.x_vis_min, params.x_vis_max, num=params.n_points, dtype=np.float64
        )

        # Y = f(x)
        y_target = func(x_grid)

        # маска для обучения и для валидации
        train_mask = (x_grid >= params.x_train_min) & (x_grid <= params.x_train_max)
        val_mask = ~train_mask

        # на всякий пожарный
        if not np.any(train_mask):
            raise ValueError(
                "В диапозон обучения каким-то образом не попала ни одна точка"
            )

        # делаем тензоры для обучения
        x_train_tensor = torch.tensor(
            x_grid[train_mask], dtype=torch.float32
        ).unsqueeze(1)
        y_train_tensor = torch.tensor(
            y_target[train_mask], dtype=torch.float32
        ).unsqueeze(1)

        # тензоры валидации
        has_val_points = bool(np.any(val_mask))
        if has_val_points:
            x_val_tensor = torch.tensor(
                x_grid[val_mask], dtype=torch.float32
            ).unsqueeze(1)
            y_val_tensor = torch.tensor(
                y_target[val_mask], dtype=torch.float32
            ).unsqueeze(1)
        else:
            x_val_tensor = None
            y_val_tensor = None

        # тензор всей зоны
        x_all_tensor = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(1)

        return PreparedData(
            x_train_tensor=x_train_tensor,
            y_train_tensor=y_train_tensor,
            has_val_points=has_val_points,
            x_val_tensor=x_val_tensor,
            y_val_tensor=y_val_tensor,
            x_all_tensor=x_all_tensor,
            x_grid=x_grid,
            y_target=y_target,
            train_mask=train_mask,
        )
