import logging
from .base_trainer import BaseTrainer
import torch
import numpy as np
from tqdm import tqdm
from src.utils.moo import circle_points


class MultiSolutionTrainer(BaseTrainer):
    def __init__(self, alpha, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.num_tasks = self.benchmark.num_tasks
        self.sampling_distribution = torch.distributions.Dirichlet(torch.ones(self.num_tasks) * self.alpha)
        self.logging_freq = 10

    def on_before_training_step(self, *args, **kwargs):
        self.ray = self.sampling_distribution.sample().cuda()
        self.model.ray = self.ray
        self.method.ray = self.ray
        return super().on_before_training_step(*args, **kwargs)

    def predict(self, test_loader, leave_tqdm=False):
        n = 11 if self.num_tasks == 2 else 66
        rays = circle_points(n, dim=self.num_tasks)
        results = {}
        for i, ray in enumerate(tqdm(rays)):
            ray = torch.from_numpy(ray.astype(np.float32)).cuda()
            ray /= ray.sum()

            self.ray = ray
            self.model.ray = self.ray
            self.method.ray = self.ray
            results[i] = super().predict(test_loader, leave_tqdm=False)

        self.present_results(results)
        for k, v in results.items():
            for kk, vv in v.items():
                name = "test/{}/ray-{}".format(kk, k)
                self.log(name, vv)

        logging.info(results)
        logging.info(self.present_results(results))
        return results

    @staticmethod
    def present_results(results):
        for k, v in results.items():
            a = ""
            for kk, vv in v.items():
                if "acc" in kk:
                    a = "{}\t{}".format(a, round(vv, 5))
                if "huber" in kk or "depth" in kk or "iou" in kk:
                    a = "{}\t{}".format(a, round(vv, 5))
            print(a)

    def _val_loop(self, leave_tqdm=False):
        if self.epoch % self.logging_freq == 0:
            n = 11 if self.num_tasks == 2 else 66
            rays = circle_points(n, dim=self.num_tasks)
            results = {}
            for i, ray in enumerate(tqdm(rays)):
                ray = torch.from_numpy(ray.astype(np.float32)).cuda()
                ray /= ray.sum()

                self.ray = ray
                self.model.ray = self.ray
                self.method.ray = self.ray
                results[i] = super()._val_loop(leave_tqdm)

            logging.info(results)
            self.present_results(results)
            for k, v in results.items():
                for kk, vv in v.items():
                    name = "val/{}/ray-{}".format(kk, k)
                    self.log(name, vv)

            logging.info(results)
            logging.info(self.present_results(results))
            return results
