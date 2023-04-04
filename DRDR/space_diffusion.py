"""
Respace the sample steps (inference evrevy X steps) to accelerate the basic Gaussian diffusion model.
Adapted from ILVR_ADM: https://github.com/jychoi118/ilvr_adm
"""

import numpy as np
import torch as th

from DRDR.diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    def __init__(self,
                 denoise_fn,
                 betas_schedule_name="linear",
                 num_timesteps=1000,
                 model_mean_type=0,
                 model_var_type=3,
                 loss_type="l2",
                 rescale_timesteps=True,
                 section=None,
                 range_t=None,
                 device="cpu"):
        super().__init__(denoise_fn, betas_schedule_name, num_timesteps,
                         model_mean_type, model_var_type, loss_type, device)
        self.rescale_timesteps = rescale_timesteps
        self.range_t = range_t
        if self.rescale_timesteps:
            assert section is not None
            self.use_timesteps = set(space_timesteps(num_timesteps, section))
            if range_t is not None:
                p = set(
                    [x for x in range(num_timesteps - range_t, num_timesteps)])
                self.use_timesteps = self.use_timesteps | p

            self.timestep_map = []

            self.original_num_steps = len(self.betas)

            # base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
            last_alpha_cumprod = 1.0
            new_betas = []
            for i, alpha_cumprod in enumerate(self.alphas_cumprod):
                if i in self.use_timesteps:
                    new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                    last_alpha_cumprod = alpha_cumprod
                    self.timestep_map.append(i)
            self.betas = np.array(new_betas)
            self.num_timesteps = int(self.betas.shape[0])

            # print(sorted(self.timestep_map))
            self.calc_paras()

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            # map_tensor = th.tensor(self.timestep_map, dtype=t.dtype)
            # new_ts = map_tensor[t]
            # new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
            # print(self.timestep_map)
            new_ts = self.timestep_map[t[0]] * (1000.0 / self.original_num_steps)
            new_ts = th.tensor([int(new_ts)] * t.shape[0], dtype=t.dtype)
            # print(new_ts)
            return new_ts.to(self.device)
        else:
            return t
