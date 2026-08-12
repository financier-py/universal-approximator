import math

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from src.data_builder import PreparedData
from src.schemas.experiment_params import ExperimentParams
from src.schemas.training_state import EpochState


class GDBTTrainer:
    """
    Движок обучения град. буста
    Генерирует кадры для анимации (EpochState) на каждой итерации добавления дерева
    """

    def __init__(self, params: ExperimentParams, data: PreparedData):
        self.params = params
        self.data = data

    def train(self):
        """Запускает бустинг. Отдает кадры анимации через yield"""
        x_train = self.data.x_grid[self.data.train_mask].reshape(-1, 1)
        y_train = self.data.y_target[self.data.train_mask]
        x_all = self.data.x_grid.reshape(-1, 1)

        # инициализируем модельку + обучаем
        gb = GradientBoostingRegressor(
            n_estimators=self.params.n_estimators,
            max_depth=self.params.max_depth,
            learning_rate=self.params.learning_rate,
            loss="squared_error",
            random_state=67,
        )
        gb.fit(x_train, y_train)

        # базовое предсказание F_0 (среднее y_train)
        f_0_train = gb.init_.predict(x_train)  # type: ignore
        f_0_grid = gb.init_.predict(x_all)  # type: ignore

        # нулевой кадр: накопленное предсказание = F_0
        y_pred_train_accum = f_0_train.copy()
        y_pred_grid_accum = f_0_grid.copy()

        # статичная матрица выходов элементов на весь процесс
        max_elements = (
            1 if self.params.gdbt_view_mode == "current" else self.params.n_estimators
        )
        current_element_outputs = np.full((max_elements, len(self.data.x_grid)), np.nan)

        yield self._create_frame(
            epoch=0,
            y_pred_grid=y_pred_grid_accum.copy(),
            element_outputs=current_element_outputs.copy(),
            y_pred_train=y_pred_train_accum.copy(),
        )

        history_contribs = []

        for m, tree_stage in enumerate(gb.estimators_, start=1):
            tree = tree_stage[0]

            tree_contrib_train = self.params.learning_rate * tree.predict(x_train)
            tree_contrib_grid = self.params.learning_rate * tree.predict(x_all)

            # накапливаем предсказание с каждым деревом
            y_pred_train_accum += tree_contrib_train
            y_pred_grid_accum += tree_contrib_grid

            if self.params.gdbt_view_mode == "current":
                raw_tree_pred = tree.predict(x_all)
                current_element_outputs = raw_tree_pred.reshape(1, -1)
            else:
                # накапливаем вклады прошлых деревьев
                history_contribs.append(tree_contrib_grid)
                current_element_outputs[: len(history_contribs)] = np.array(
                    history_contribs
                )

            # отдаем кадр в UI по шагам
            if m % self.params.yield_every == 0 or m == self.params.n_estimators:
                yield self._create_frame(
                    epoch=m,
                    y_pred_grid=y_pred_grid_accum.copy(),
                    element_outputs=current_element_outputs.copy(),
                    y_pred_train=y_pred_train_accum.copy(),
                )

    def _create_frame(
        self,
        epoch: int,
        y_pred_grid: np.ndarray,
        element_outputs: np.ndarray,
        y_pred_train: np.ndarray,
    ) -> EpochState:
        y_train_target = self.data.y_target[self.data.train_mask]

        # RMSE Train
        mse_train = np.mean((y_train_target - y_pred_train) ** 2)
        rmse_train = math.sqrt(mse_train)

        # RMSE Val
        if self.data.has_val_points:
            y_val_target = self.data.y_target[~self.data.train_mask]
            y_val_pred = y_pred_grid[~self.data.train_mask]
            mse_val = np.mean((y_val_target - y_val_pred) ** 2)
            rmse_val = math.sqrt(mse_val)
        else:
            rmse_val = 0.0

        return EpochState(
            model_type="gdbt",
            epoch=epoch,
            loss_train=rmse_train,
            loss_val=rmse_val,
            x_grid=self.data.x_grid,
            y_target=self.data.y_target,
            y_pred=y_pred_grid,
            element_outputs=element_outputs,
            train_mask=self.data.train_mask,
        )
