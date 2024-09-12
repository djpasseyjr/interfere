"""Agent based model of fish in an aquarium coupled to a simulation of the
nitrogen cycle.

Taken from:
https://github.com/jacobsinclair/Aquaculture/blob/c8ec9e6c852531a4b7d38f55ef65ce27aeb28967/aquarium.py

See also: Fazio et. al. (2006) Mathematical and numerical modeling for a
bio-chemical aquarium.
"""
from typing import Optional, Union

import numpy as np

from .base import StochasticDifferentialEquation
from ..utils import copy_doc


class PlantedTankNitrogenCycle(StochasticDifferentialEquation):
    
    def __init__(
        self,
        reference_time_v: float = 6 * 7 * 24 * 60 * 60.0,  
        t_min: float = 0.0,
        ammonia_eff_gamma: float = 5e-6,
        amminia_max_c1: float = 5.0,
        ammonia_eff_x_min: float = 0.0,
        ammonia_eff_x_max: float = 24 * np.pi,
        oxy_prod_p2: float = 1e-4,
        oxy_max_c2: float = 8.5,
        nitrite_cons_p3: float = 0.0,
        nitrate_cons_p4: float = 2e-7,
        nitrite_bact_eff_gamma1: float = 2e-8,
        nitrite_bact_eff_y_min: float = -6 * np.pi,
        nitrite_bact_eff_y_max: float = 5 * np.pi,
        nitrate_bact_eff_gamma2: float = 3e-6,
        nitrate_bact_eff_z_min: float = -6 * np.pi,
        nitrate_bact_eff_z_max: float = 2 * np.pi,
        sigma: Union[float, np.ndarray] = 0,
        measurement_noise_std: Optional[np.ndarray] = None,
    ):
        """Initializes a simulation of the nitrogen cycle in a planted tank.

        Model describes the chemical reaction

        2NH4+ + 3O2 -> 2N02- + 2H2O + 4H+
        2N02- + O2 -> 2NO3-

        where efficiency and production rates are governed by populations of
        bacteria and plant life. Bacteria and plant growth are modeled
        as exogenous functions of time. These functions are defined in
        `ammon_prod_eff_p1`, `nitrite_bacteria_effic_mu1`, and
        `nitrate_bacteria_effic_mu2`.
        
        Parameter names from the original paper have been preserved at the end
        of variables.

        Args:
            reference_time_v (float): How long it takes the plants and bacteria
                to grow to full capacity.
            t_min (float): Start time for bacteria growth.
            ammonia_eff_gamma (float): Parameter that scales the ammonia
            production efficiency. Relates to the amount of uneaten fish food,
                and fish urine and feces.
            amminia_max_c1 (float): Max healthy concentration of ammonia. 
            ammonia_eff_x_min (float): Parameter that controls the inital level
                of ammonia production efficiency. Relates to the amount of
                uneaten fish food, and fish urine and feces.
            ammonia_eff_x_max (float): Parameter that controls the inital level
                of ammonia production efficiency. Relates to the amount of
                uneaten fish food, and fish urine and feces.
            oxy_prod_p2 (float): Oxygen production efficency---this is constant.
                Relates to the amount of plants in the tank.
            oxy_max_c2 (float): Max oxygen level.
            nitrite_cons_p3 (float): Rate of nitrite consumption. Model
                constant. 
            nitrate_cons_p4 (float): Rate of nitrate consumption. Model
                constant. 
            nitrite_bact_eff_gamma1 (float): Parameter that scales the nitrite
                production efficiency. Relates to nitrosomonous
                and nitrosospira bacteria.
            nitrite_bact_eff_y_min (float): Parameter that controls the inital
                level of nitrite production efficiency. Relates to nitrosomonous
                and nitrosospira bacteria.
            nitrite_bact_eff_y_max (float): Parameter that controls the final
                level of nitrate production efficiency. Relates to nitrosomonous
                and nitrosospira bacteria.
            nitrate_bact_eff_gamma2 (float): Parameter that scales the nitrate
                production efficiency. Relates to the nitrobacter and nitrospira bacteria.
            nitrate_bact_eff_z_min (float): Parameter that controls the inital
                level of nitrate production efficiency. Relates to the nitrobacter and nitrospira bacteria.
            nitrate_bact_eff_z_max (float): Parameter that controls the final
                level of nitrate production efficiency. Relates to the nitrobacter and nitrospira bacteria.
            sigma (float or ndarray): The stochastic noise parameter. Can be a
                float, a 1D matrix or a 2D matrix. Dimension must match
                dimension of model.
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, if the dynamic model had two variables x1
                and x2 and `measurement_noise_std = [1, 10]`, then
                independent gaussian noise with standard deviation 1 and 10
                will be added to x1 and x2 respectively at each point in time. 

        References:
            Fazio et. al. (2006) Mathematical and numerical modeling for a
            bio-chemical aquarium. 
        """        
        self.ammonia_limit_q1 = 1 / amminia_max_c1
        ammonia_eff_const = np.arctan(ammonia_eff_x_min)

        self.oxy_limit_q2 = 1 / oxy_max_c2

        nitirite_bact_eff_const = np.arctan((nitrite_bact_eff_y_max - nitrite_bact_eff_y_min) * t_min / reference_time_v + nitrite_bact_eff_y_min)

        nitrate_bact_eff_const = np.arctan((nitrate_bact_eff_z_max - nitrate_bact_eff_z_min) * t_min / reference_time_v + nitrate_bact_eff_z_min)

        self.oxy_prod_p2 = oxy_prod_p2
        self.nitrite_cons_p3 = nitrite_cons_p3
        self.nitrate_cons_p4 = nitrate_cons_p4

        # p1 in paper.
        self.ammon_prod_eff_p1 = lambda t: ammonia_eff_gamma * (np.arctan((ammonia_eff_x_max - ammonia_eff_x_min) * t / reference_time_v + ammonia_eff_x_min) - ammonia_eff_const) 

        # mu1 in the paper. 
        self.nitrite_bacteria_effic_mu1 = lambda t: nitrite_bact_eff_gamma1 * (np.arctan((nitrite_bact_eff_y_max - nitrite_bact_eff_y_min) * t / reference_time_v + nitrite_bact_eff_y_min) - nitirite_bact_eff_const)

        # mu2 in paper.
        self.nitrate_bacteria_effic_mu2 = lambda t: nitrate_bact_eff_gamma2 * (np.arctan((nitrate_bact_eff_z_max - nitrate_bact_eff_z_min) * t / reference_time_v + nitrate_bact_eff_z_min) - nitrate_bact_eff_const)

        dim = 4
        super().__init__(dim, measurement_noise_std, sigma)


    @copy_doc(StochasticDifferentialEquation.drift)
    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        ammonia, oxy, nitrite, nitrate = x
    
        d_ammonium_dt = self.ammon_prod_eff_p1(t) * (1 - self.ammonia_limit_q1 * ammonia) - self.nitrite_bacteria_effic_mu1(t) * ammonia **2 * oxy ** 3

        d_oxy_dt = self.oxy_prod_p2 * (1 - self.oxy_limit_q2 * oxy ) - self.nitrite_bacteria_effic_mu1(t) * ammonia **2 * oxy ** 3 - self.nitrate_bacteria_effic_mu2(t) * oxy ** 0.5 * nitrite

        d_nitrite_dt = self.nitrite_bacteria_effic_mu1(t) * ammonia **2 * oxy ** 3 - self.nitrate_bacteria_effic_mu2(t) * oxy ** 0.5 * nitrite - self.nitrite_cons_p3 * nitrite

        d_nitrate_dt = self.nitrate_bacteria_effic_mu2(t) * oxy ** 0.5 * nitrite - self.nitrate_cons_p4 * nitrate

        return np.array([d_ammonium_dt, d_oxy_dt, d_nitrite_dt, d_nitrate_dt])
    

    @copy_doc(StochasticDifferentialEquation.noise)
    def noise(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.sigma
