from abc import ABC
from enum import Enum


class TrainerMode(Enum):
    UNDEFINED = None
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class TrainerStateManagerMixin(ABC):
    current_step: int = 0
    current_epoch: int = 0
    current_mode: TrainerMode = TrainerMode.UNDEFINED

    def _set_train(self):
        self.current_mode = TrainerMode.TRAIN

    def _set_val(self):
        self.current_mode = TrainerMode.VAL

    def _set_test(self):
        self.current_mode = TrainerMode.TEST

    def _tick_step(self):
        self.current_step += 1

    def _tick_epoch(self):
        self.current_epoch += 1

    def tick(self, interval: str):
        if interval == "step":
            self._tick_step()
        elif interval == "epoch":
            self._tick_epoch()
        else:
            raise ValueError("Supported intervals are 'step' and 'epoch'")
