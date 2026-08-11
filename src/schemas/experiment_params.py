from typing import Literal

from pydantic import BaseModel, Field, model_validator

# допустимые модели
ModelType = Literal["nn", "gdbt"]

# допустимые ф-ии активации
ActivationType = Literal["sigmoid", "relu", "tanh", "gelu", "leaky_relu"]


class ExperimentParams(BaseModel):
    """
    Конфигурация параметров обучения и визуализации.

    Тут настройки мат. функций, структуры сети,
    диапозоны для обучения и визуализации и гиперпараметры модели.
    """

    # выбор алгоритма
    model_type: ModelType = Field(
        default="nn", description="Тип модели 'nn' - нейронка, 'ddbt' - град. буст"
    )

    # мат. функция и сетка точек
    expression: str = Field(
        default="sin(x) * exp(-0.1 * x)",
        description="f(x) введенная пользователем",
        min_length=1,
    )
    n_points: int = Field(
        default=500, ge=50, le=5000, description="Кол-во точек для построение графика"
    )

    # диапозоны обучения + визуализации
    x_vis_min: float = Field(
        default=-5.0, description="Мин. значение x для общего диапозона"
    )
    x_vis_max: float = Field(
        default=5.0, description="Макс. значение x для общего диапозона"
    )
    x_train_min: float = Field(default=-2.0, description="Мин. значение x для обучения")
    x_train_max: float = Field(default=2.0, description="Макс. значение x для обучения")

    # архитектура НЕЙРОСЕТИ
    n_neurons: int = Field(
        default=10, ge=1, le=500, description="Кол-во нейронов в скрытом слое"
    )
    activation_type: ActivationType = Field(
        default="sigmoid", description="Ф-ия активации скрытого слоя"
    )

    # параметры ГРАД БУСТА
    n_estimators: int = Field(default=100, ge=1, le=2000, description="Кол-во деревьев")
    max_depth: int = Field(default=2, ge=1, le=2000, description="Макс. глубина дерева")

    # гиперпараметры обучения
    learning_rate: float = Field(
        default=0.01, gt=0.0, le=1.0, description="Скорость обучения"
    )
    epochs: int = Field(
        default=100, ge=1, le=20_000, description="Кол-во эпох обучения"
    )
    yield_every: int = Field(
        default=5, ge=1, description="Частота отдачи кадров в UI (каждую N эпоху)"
    )

    @model_validator(mode="after")
    def validate_ranges(self) -> "ExperimentParams":
        # min должен быть меньшу max
        if self.x_vis_max <= self.x_vis_min:
            raise ValueError(
                f"x_vis_min ({self.x_vis_min}) должен быть меньше x_vis_max ({self.x_vis_max})."
            )
        if self.x_train_max <= self.x_train_min:
            raise ValueError(
                f"x_train_min ({self.x_train_min}) должен быть меньше x_train_max ({self.x_train_max})."
            )

        # диапозон обучения ОБЯЗАН БЫТЬ внутри диапозона визуализации
        if self.x_train_min < self.x_vis_min or self.x_train_max > self.x_vis_max:
            raise ValueError(
                f"Диапазон обучения [{self.x_train_min}, {self.x_train_max}] "
                f"должен находиться внутри диапазона визуализации [{self.x_vis_min}, {self.x_vis_max}]."
            )

        return self
