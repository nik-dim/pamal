from rotograd import RotoGrad

from src.models.base_model import BaseModel


class RotogradWrapper(RotoGrad, BaseModel):
    pass

    def forward(self, x, *args, **kwargs):
        return super().forward(x), []

    def shared_parameters(self):
        return []

    def task_specific_parameters(self):
        return []

    def last_shared_parameters(self):
        return []
