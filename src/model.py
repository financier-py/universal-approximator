from dataclasses import dataclass

import torch
from torch import nn

from activations import get_activation_function
from schemas.experiment_params import ExperimentParams


@dataclass(frozen=True)
class NetworkOutput:
    """Результат прямого прохода сети"""

    y_pred: torch.Tensor  # (K, 1)
    neuron_outputs: torch.Tensor  # (N, K) - вклад каждого из N нейронов


class FunctionApproximationNet(nn.Module):
    """
    MLP с одним скрытым слоем
    При forward проходе выдает не только итоговый Y
    но и вклады каждого нейрона
    """

    def __init__(self, params: ExperimentParams):
        super().__init__()

        self.n_neurons = params.n_neurons
        self.hidden_layer = nn.Linear(in_features=1, out_features=self.n_neurons)
        self.activation = get_activation_function(params.activation_type)
        self.output_layer = nn.Linear(in_features=self.n_neurons, out_features=1)

    def forward(self, x: torch.Tensor):
        """Прямой проход с декомпозицией каждого нейрона"""

        z = self.hidden_layer(x)
        a = self.activation(z)

        w_out = self.output_layer.weight
        neuron_contributes = a * w_out
        neuron_contributes = neuron_contributes.T

        y_pred = self.output_layer(a)

        return NetworkOutput(y_pred=y_pred, neuron_outputs=neuron_contributes)
