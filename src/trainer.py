import math

import torch
import torch.nn as nn  # noqa: PLR0402
import torch.optim as optim  # noqa: PLR0402

from data_builder import PreparedData
from model import FunctionApproximationNet
from schemas.experiment_params import ExperimentParams
from schemas.training_state import EpochState


class Trainer:
    """
    Движок обучения нейросети. Отдает свое состояние каждый N эпох
    """

    def __init__(self, params: ExperimentParams, data: PreparedData):
        self.params = params
        self.data = data

        self.model = FunctionApproximationNet(params)
        self.optimizer = optim.Adam(self.model.parameters(), lr=params.learning_rate)
        self.criterion = nn.MSELoss()

    def train(self):
        """Запускает цикл обучения и отдает кадры анимации через yield"""
        # нулевой кадр
        yield self._capture_frame(epoch=0)

        for epoch in range(1, self.params.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()

            # прямой проход. учимся только на train
            output = self.model(self.data.x_train_tensor)
            loss_train = self.criterion(output.y_pred, self.data.y_train_tensor)

            # обратный проход
            loss_train.backward()
            self.optimizer.step()

            # отдаем кадр в UI
            if epoch % self.params.yield_every == 0 or epoch == self.params.epochs:
                yield self._capture_frame(epoch=epoch)

    def _capture_frame(self, epoch: int):
        """Делает кадр состояния модельки и считает метрики"""
        self.model.eval()

        with torch.no_grad():
            # Считаем RMSE на обучающем интервале
            train_output = self.model(self.data.x_train_tensor)
            loss_train = self.criterion(
                train_output.y_pred, self.data.y_train_tensor
            ).item()
            rmse_train = math.sqrt(loss_train)

            # Считаем RMSE на валидации
            if (
                self.data.has_val_points
                and self.data.x_val_tensor is not None
                and self.data.y_val_tensor is not None
            ):
                val_output = self.model(self.data.x_val_tensor)
                loss_val = self.criterion(
                    val_output.y_pred, self.data.y_val_tensor
                ).item()
                rmse_val = math.sqrt(loss_val)
            else:
                rmse_val = 0.0

            # прогон всей сетки
            full_output = self.model(self.data.x_all_tensor)
            y_pred_np = full_output.y_pred.squeeze(1).cpu().numpy()
            neuron_outputs_np = full_output.neuron_outputs.cpu().numpy()

        return EpochState(
            epoch=epoch,
            loss_train=rmse_train,
            loss_val=rmse_val,
            x_grid=self.data.x_grid,
            y_target=self.data.y_target,
            y_pred=y_pred_np,
            neuron_outputs=neuron_outputs_np,
            train_mask=self.data.train_mask,
        )
