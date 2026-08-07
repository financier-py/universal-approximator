import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class EpochState(BaseModel):
    """
    Состояние обучения на определенной эпохе.
    Вся нужная инфа для построения графиков в UI.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    epoch: int = Field(..., ge=0, description="Номер текущей эпохи")
    loss: float = Field(..., description="Текущая ошибка MSE (только на train)")

    # массивы для графиков
    x_grid: np.ndarray = Field(
        ..., description="Точки X на всем диапозоне визуализации (1D array)"
    )
    y_target: np.ndarray = Field(
        ..., description="Значения целевой ф-ии на x_grid (1D array)"
    )
    y_pred: np.ndarray = Field(
        ..., description="Предсказания модели на x_grid (1D array)"
    )
    neuron_outputs: np.ndarray = Field(
        ..., description="Вклад каждого из N нейронов на сетке x_grid (2D array)"
    )
    train_mask: np.ndarray = Field(
        ...,
        description="1D булев массив. True значит, что точка входит в обучающий диапозон.",
    )
