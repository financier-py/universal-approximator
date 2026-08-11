import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.schemas.experiment_params import ModelType


class EpochState(BaseModel):
    """
    Состояние обучения на определенной эпохе.
    Вся нужная инфа для построения графиков в UI.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_type: ModelType = Field(
        default="nn", description="Тип модели, сформировавший кадр"
    )

    epoch: int = Field(..., ge=0, description="Номер текущей эпохи или дерева")
    loss_train: float = Field(..., description="RMSE на обучающем интервале")
    loss_val: float = Field(..., description="RMSE вне обучающего интервала")

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
    element_outputs: np.ndarray = Field(
        ...,
        description=(
            "Вклады базовых элементов на x_grid\n"
            "Для NN: (N, K) - вклады каждого из N нейронов\n"
            "Для GBDT current: (1, K) - последнее дерево без масштабирования\n"
            "Для GBDT history: (M_history, K) - сумма вкладов всех пред. деревьев"
        ),
    )
    train_mask: np.ndarray = Field(
        ...,
        description="1D булев массив. True значит, что точка входит в обучающий диапозон.",
    )
