from elasticdeform.deform_grid import (
    _normalize_axis_list,
    _normalize_inputs,
    deform_grid,
)
import numpy as np


class ElasticDeformation:
    def __init__(self, points=2, sigma=5, axis=[0, 1, 2]):

        points_disp = [points] * 3
        points_grid = [2 + points] * 3
        self.displacement = np.zeros((3, *points_grid))
        for dsp in self.displacement:
            dsp[1:-1, 1:-1, 1:-1] = np.random.randn(*points_disp) * sigma

    def deform(self, data, order):
        assert (
            data.ndim == 3 or data.ndim == 4
        ), "Data dimension should be either 3 or 4."
        axis = (0, 1, 2) if data.ndim == 3 else (1, 2, 3)

        data_t = deform_grid(
            data, self.displacement, axis=axis, order=order, mode="nearest"
        )
        return data_t
