import torch.nn as nn  # noqa: PLR0402

from schemas.experiment_params import ActivationType


def get_activation_function(activation_type: ActivationType) -> nn.Module:
    """Возвращает ф-ию активации по ее имени"""

    activations = {
        "sigmoid": nn.Sigmoid(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(),
    }

    if activation_type not in activations:
        raise ValueError(
            f"Ф-ия активации {activation_type} недопустима :( "
            f"Вот допустимые варианты: {list(activations.keys())}"
        )

    return activations[activation_type]
