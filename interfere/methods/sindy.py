import numpy as np
import pysindy as ps
from sklearn.base import BaseEstimator

from interfere.utils import copy_doc


class SINDY(ps.SINDy, BaseEstimator):

    @copy_doc(ps.SINDy.__init__)
    def __init__(self, 
        optimizer=None,
        feature_library=None,
        differentiation_method=None,
        feature_names=None,
        t_default=1,
        discrete_time=False,
        **kwargs
    ):
        super().__init__(
            optimizer, feature_library, differentiation_method, feature_names, t_default, discrete_time,
        )
        # self.sindy_model.set_params(**kwargs)


    def simulate_counterfactual(
            self, X, time_points, intervention, max_sim_val=1e6):
        """Fits and simulates an intervention."""

        # Pass the intervention variable as a control signal
        endo_X, exog_X = intervention.split_exogeneous(X)
        self.fit(endo_X, t=time_points, u=exog_X)

        # Create a perfect intervention control signal
        interv_X = intervention.constants * np.ones_like(exog_X)

        # Initial condition (Exogenous signal removed.)
        x0 = endo_X[0]

        # Sindy uses scipy.integrate.solve_ivp by default and solve_ivp
        # uses event functions with assigned attributes as callbacks.
        # The below code tells scipy to stop integrating when
        # too_big(t, y) == True.
        too_big = lambda t, y: np.all(np.abs(y) < max_sim_val)
        too_big.terminal = True

        # Simulate with intervention
        sindy_X_do = self.simulate(
            x0, time_points, u=interv_X, integrator_kws={"events": too_big})

        # Retrive number of successful steps
        n_steps = sindy_X_do.shape[0]
        
        # Reassemble the control and response signals into a single array
        pred_X_do = intervention.combine_exogeneous(
            sindy_X_do, interv_X[:n_steps])
        
        return pred_X_do
        

def sindy_perf_interv_extrapolate(
    X,
    t,
    intervention_idx,
    intervention_value,
    max_sim_val=1e6,
):
    """Predicts the effect of a perfect intervention on the observed system.
    """
    # Pass the intervention variable as a control signal
    u = X[:, intervention_idx]
    trainingX = np.delete(X, intervention_idx, axis=1)
    method = ps.SINDy()
    method.fit(trainingX, t=t, u=u)

    # Create a perfect intervention control signal
    sindy_interv = intervention_value * np.ones_like(u)

    # Initial condition (Remove control signal)
    x0 = np.delete(X[0, :], intervention_idx)

    # Sindy uses scipy.integrate.solve_ivp by default and solve_ivp
    # uses event functions with assigned attributes as callbacks.
    # The below code tells scipy to stop integrating when
    # too_big(t, y) == True.
    too_big = lambda t, y: np.all(np.abs(y) < max_sim_val)
    too_big.terminal = True

    # Simulate with intervention
    sindy_X_do = method.simulate(
        x0, t, u=sindy_interv, integrator_kws={"events": too_big})

    # Retrive number of successful steps
    n_steps = sindy_X_do.shape[0]
    # Reassemble the control and response signals into a single array
    sindy_all_sigs = np.insert(
        sindy_X_do, intervention_idx, sindy_interv[:n_steps], axis=1)
    
    return sindy_all_sigs