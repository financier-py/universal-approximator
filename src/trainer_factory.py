from src.data_builder import PreparedData
from src.gdbt_trainer import GDBTTrainer
from src.nn_trainer import NNTrainer
from src.schemas.experiment_params import ExperimentParams


def get_trainer(params: ExperimentParams, data: PreparedData):
    """
    Фабрика для создания трейнера в зависимости от выбранного типа модели (NN или gdbt)
    """
    if params.model_type == "gdbt":
        return GDBTTrainer(params, data)
    return NNTrainer(params, data)
