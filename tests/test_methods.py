import re
from typing import Type

import interfere
import interfere._methods
import interfere._methods.sindy
from interfere._methods.deep_learning import LTSF
import interfere.methods
import numpy as np
import pytest

# Check for optuna installation.
optuna_installed = True
try:
    import optuna
except ImportError:
    optuna_installed = False

SEED = 11
PRED_LEN = 10
OPTUNA_NTRIALS = 6
MAX_SECS_PER_OPTUNA_TRIAL = 180
REQUIRED_SUCCESSFUL_OPTUNA_TRIALS = 3

METHODS = [
    interfere.methods.ARIMA,
    interfere.methods.AverageMethod,
    interfere.methods.ResComp,
    interfere.methods.SINDy, 
    LTSF,
    interfere.methods.VAR,
    interfere.methods.LSTM,
    interfere.methods.NHITS,
]
EXOG_RESP_METHODS = []


def VARIMA_timeseries(dim=5, lags=3, noise_lags=2, tsteps=100, n_do=20):
    """Simulates a VARIMA time series.

    Args:
        dim: The number of entries in the state vector.
        lags: Number of time lags in model.
        noise lags: Number of lags in noise autocorrelation.
        tsteps: Time steps to simulate.
        n_do: Number of time steps of intervention response to simulate.

    Returns:
        historic_X: Array of historic states with shape (tsteps, dim).
        historic_t: Array of time values with shape (tsteps,)
        X_do: Array of intervention response states with shape (n_do, dim).
        t_do: Array of time values corresponding to intervention response with
            shape (n_do,)
        intervention (interfere.interventions.ExogIntervention): The
            intervention applied. 
    """

    # Initialize a VARMA model
    rng = np.random.default_rng(SEED)
    phis = [0.5 * (rng.random((dim, dim)) - 0.5) for i in range(lags)]
    thetas = [0.5 * (rng.random((dim, dim)) - 0.5) for i in range(noise_lags)]
    sigma = 0.5 * np.eye(dim)
    model = interfere.dynamics.VARMADynamics(phis, thetas, sigma)

    max_lags = max(lags, noise_lags)
    # Generate a time series
    t = np.arange(tsteps)
    prior_states = rng.random((max_lags, dim))
    X = model.simulate(t, prior_states, rng=rng)

    intervention = interfere.PerfectIntervention([0, 1], [-0.5, -0.5])
    historic_t = t[:-n_do]
    historic_X = X[:-n_do, :]
    t_do = t[(-n_do - 1):]

    X_do = model.simulate(
        t_do,
        historic_X,
        prior_t=historic_t,
        intervention=intervention,
        rng=rng
    )

    return historic_X, historic_t, X_do, t_do, intervention

def belozyorov_timeseries(tsteps=100, n_do=20):
    """Simulates a VARIMA time series.

    Args:
        tsteps: Time steps to simulate.
        n_do: Number of time steps of intervention response to simulate.

    Returns:
        historic_X: Array of historic states with shape (tsteps, dim).
        historic_t: Array of time values with shape (tsteps,)
        X_do: Array of intervention response states with shape (n_do, dim).
        t_do: Array of time values corresponding to intervention response with
            shape (n_do,)
        intervention (interfere.interventions.ExogIntervention): The
            intervention applied. 
    """
    rng = np.random.default_rng(SEED)
    lags = 1
    dim = 3
    model = interfere.dynamics.Belozyorov3DQuad(
        mu=1.81, sigma=0.05, measurement_noise_std = 0.01 * np.ones(dim),
    )

    # Generate a time series
    t = np.linspace(0, 1, tsteps)
    x0 = rng.random(dim)
    X = model.simulate(t, x0, rng=rng)

    intervention = interfere.PerfectIntervention(0, 5.0)
    historic_times = t[:-n_do]
    X_historic = X[:-n_do, :]
    forecast_times = t[(-n_do - 1):]
    X0_do = X_historic[-lags, :]

    X_do = model.simulate(
        forecast_times,
        X0_do,
        intervention=intervention,
        rng=rng
    )

    return X_historic, historic_times, X_do, forecast_times, intervention


def make_time_series_combos(dynamics):
    X_historic, prior_t, X_do, forecast_times, intervention = dynamics

    # Create time series combonations.
    prior_endog_states, prior_exog_states = intervention.split_exog(X_historic)
    endo_true, exog = intervention.split_exog(X_do)
    forecast_times = forecast_times[:PRED_LEN]
    endo_true = endo_true[:PRED_LEN, :]
    exog = exog[:PRED_LEN, :]

    return prior_endog_states, prior_exog_states, prior_t, endo_true, exog, forecast_times, intervention


def make_optuna_training_data():
    # Generate sample data.
    model = interfere.dynamics.Belozyorov3DQuad()
    dt = 0.05
    train_t = np.arange(0, 1.5 + dt, dt)
    forecast_t = np.arange(1.5, 2 + dt, dt)
    prior_states = np.random.rand(2, model.dim)

    # Train data.
    train_states = model.simulate(train_t, prior_states)

    # Forecast data.
    forecast_states = model.simulate(
        t=forecast_t, prior_states=train_states)
    
    # Intervention data.
    intervention = interfere.PerfectIntervention(0, 1.0)
    interv_states = model.simulate(
        t=forecast_t, 
        prior_states=train_states, 
        intervention=intervention
    )

    return (
        train_t, train_states, forecast_t, 
        forecast_states, interv_states, intervention
    )


OPTUNA_TRAINING_DATA = make_optuna_training_data()
DYNAMICS = [
    make_time_series_combos(VARIMA_timeseries()),
    make_time_series_combos(belozyorov_timeseries())
]

@pytest.mark.parametrize("method_type", METHODS)
@pytest.mark.parametrize("dynamics", DYNAMICS)
class TestFitPredict:


    def test_fit(self, method_type: Type[interfere.ForecastMethod], dynamics):
        """Tests fit() method without exogenous data.
        
        Args:
            method_type: An interfere.ForecastMethod
                method type.
            dynamics: A tuple containing
                prior_endog_states: An array of historic endogenous states.
                prior_exog_states: An array of historic exogenous states.
                prior_t: Time points corresponing to the rows of the two arrays
                    above. 
                endo_true: An array of endog states for forecasting.
                exog: An array of exog states to be used in forecasting.
                forecast_times: Times corresponidng to the rows of the two
                    arrays above.
                intervention: An instance of
                    interfere.interventions.ExogIntervention.
        
        Note:
            Args are passed via decorators for the test class.
        """
        (prior_endog_states, prior_exog_states, prior_t, endo_true, 
         exog, forecast_times, intervention) = dynamics
        
        method_params = method_type.get_test_params()

        # Test fit without exog.
        method = method_type(**method_params)
        method.fit(prior_t, prior_endog_states, None)

        assert method.is_fit
        


    def test_fit_exog(self, method_type: Type[interfere.ForecastMethod], dynamics):
        """Tests fit() method with exogenous data.

        Args:
            method_type: An interfere.ForecastMethod
                method type.
            dynamics: A tuple containing
                prior_endog_states: An array of historic endogenous states.
                prior_exog_states: An array of historic exogenous states.
                prior_t: Time points corresponing to the rows of the two arrays
                    above. 
                endo_true: An array of endog states for forecasting.
                exog: An array of exog states to be used in forecasting.
                forecast_times: Times corresponidng to the rows of the two
                    arrays above.
                intervention: An instance of
                    interfere.interventions.ExogIntervention.
        
        Note:
            Args are passed via decorators for the test class.
        """

        (prior_endog_states, prior_exog_states, prior_t, endo_true, 
         exog, forecast_times, intervention) = dynamics
        
        method_params = method_type.get_test_params()

        # Test fit with exog.
        method = method_type(**method_params)
        method.fit(prior_t, prior_endog_states, prior_exog_states)

        assert method.is_fit


    def test_simulate(self, method_type: Type[interfere.ForecastMethod], dynamics):
        """Tests the simulate() method with exogenous data.
        
        Args:
            method_type: An interfere.ForecastMethod
                method type.
            dynamics: A tuple containing
                prior_endog_states: An array of historic endogenous states.
                prior_exog_states: An array of historic exogenous states.
                prior_t: Time points corresponing to the rows of the two arrays
                    above. 
                endo_true: An array of endog states for forecasting.
                exog: An array of exog states to be used in forecasting.
                forecast_times: Times corresponidng to the rows of the two
                    arrays above.
                intervention: An instance of
                    interfere.interventions.ExogIntervention.
        
        Note:
            Args are passed via decorators for the test class.
        """

        (prior_endog_states, prior_exog_states, prior_t, endo_true, 
        exog, forecast_times, intervention) = dynamics
        
        method_params = method_type.get_test_params()

        # Test simulate with exog.
        X_historic = intervention.combine_exog(
            prior_endog_states, prior_exog_states)
        X_do = intervention.combine_exog(endo_true, exog)

        method = method_type(**method_params)
        method.fit(prior_t, prior_endog_states,  prior_exog_states)
        # Simulate without exog
        method.fit(prior_t, X_historic)
        X_do_pred = method.simulate(
            t=forecast_times,
            prior_states=X_historic,
            prior_t=prior_t,
        )

        assert X_do_pred.shape == (PRED_LEN, X_do.shape[1])


    def test_simulate_ident(self, method_type: Type[interfere.ForecastMethod], dynamics):
        """Tests the simulate() method with exogenous data.
        
        Args:
            method_type: An interfere.ForecastMethod
                method type.
            dynamics: A tuple containing
                prior_endog_states: An array of historic endogenous states.
                prior_exog_states: An array of historic exogenous states.
                prior_t: Time points corresponing to the rows of the two arrays
                    above. 
                endo_true: An array of endog states for forecasting.
                exog: An array of exog states to be used in forecasting.
                forecast_times: Times corresponidng to the rows of the two
                    arrays above.
                intervention: An instance of
                    interfere.interventions.ExogIntervention.
        
        Note:
            Args are passed via decorators for the test class.
        """

        (prior_endog_states, prior_exog_states, prior_t, endo_true, 
        exog, forecast_times, intervention) = dynamics
        
        method_params = method_type.get_test_params()

        # Test simulate with exog.
        X_historic = intervention.combine_exog(
            prior_endog_states, prior_exog_states)
        X_do = intervention.combine_exog(endo_true, exog)

        method = method_type(**method_params)
        method.fit(prior_t, prior_endog_states,  prior_exog_states)
        # Simulate without exog
        method.fit(prior_t, X_historic)
        X_do_pred = method.simulate(
            t=forecast_times,
            prior_states=X_historic,
            prior_t=prior_t,
            intervention=interfere.interventions.IdentityIntervention(),
        )

        assert X_do_pred.shape == (PRED_LEN, X_do.shape[1])
            

    def test_simulate_exog(self, method_type: Type[interfere.ForecastMethod], dynamics):
        """Tests the simulate() method with exogenous data.
        
        Args:
            method_type: An interfere.ForecastMethod
                method type.
            dynamics: A tuple containing
                prior_endog_states: An array of historic endogenous states.
                prior_exog_states: An array of historic exogenous states.
                prior_t: Time points corresponing to the rows of the two arrays
                    above. 
                endo_true: An array of endog states for forecasting.
                exog: An array of exog states to be used in forecasting.
                forecast_times: Times corresponidng to the rows of the two
                    arrays above.
                intervention: An instance of
                    interfere.interventions.ExogIntervention.
        
        Note:
            Args are passed via decorators for the test class.
        """

        (prior_endog_states, prior_exog_states, prior_t, endo_true, 
         exog, forecast_times, intervention) = dynamics
        
        method_params = method_type.get_test_params()

        # Test simulate with exog.
        X_historic = intervention.combine_exog(
            prior_endog_states, prior_exog_states)
        X_do = intervention.combine_exog(endo_true, exog)

        method = method_type(**method_params)
        method.fit(prior_t, prior_endog_states,  prior_exog_states)
        X_do_pred = method.simulate(
                t=forecast_times,
                prior_states=X_historic,
                prior_t=prior_t,
                intervention=intervention,
            )

        assert X_do_pred.shape == (PRED_LEN, X_do.shape[1])

        # Make sure simulate returns the exogenous data correctly.
        exog_true = intervention.eval_at_times(forecast_times)
        _, exog_pred = intervention.split_exog(X_do_pred)
        assert np.allclose(exog_true, exog_pred), (
            "Exogeneous signal not close to expected.\n\n"
            f"Target = \n{exog_true}"
            f"\nPredicted = \n{exog_pred}"
        )


    def test_predict(self, method_type: Type[interfere.ForecastMethod], dynamics):
        """Tests the predict() method without exogenous data.
        
        Args:
            method_type: An interfere.ForecastMethod
                method type.
            dynamics: A tuple containing
                prior_endog_states: An array of historic endogenous states.
                prior_exog_states: An array of historic exogenous states.
                prior_t: Time points corresponing to the rows of the two arrays
                    above. 
                endo_true: An array of endog states for forecasting.
                exog: An array of exog states to be used in forecasting.
                forecast_times: Times corresponidng to the rows of the two
                    arrays above.
                intervention: An instance of
                    interfere.interventions.ExogIntervention.
        
        Note:
            Args are passed via decorators for the test class.
        """

        (prior_endog_states, prior_exog_states, prior_t, endo_true, 
         exog, forecast_times, intervention) = dynamics
        
        method_params = method_type.get_test_params()
        method = method_type(**method_params)

        method.fit(prior_t, prior_endog_states)

        endo_pred = method.predict(
            forecast_times,
            prior_endog_states,
            prior_t=prior_t
        )

        assert endo_pred.shape[1] == endo_true.shape[1]
        assert endo_pred.shape[0] == PRED_LEN



    def test_predict_exog(self, method_type: Type[interfere.ForecastMethod], dynamics):
        """Tests the predict() method with exogenous data.
        
        Args:
            method_type: An interfere.ForecastMethod
                method type.
            dynamics: A tuple containing
                prior_endog_states: An array of historic endogenous states.
                prior_exog_states: An array of historic exogenous states.
                prior_t: Time points corresponing to the rows of the two arrays
                    above. 
                endo_true: An array of endog states for forecasting.
                exog: An array of exog states to be used in forecasting.
                forecast_times: Times corresponidng to the rows of the two
                    arrays above.
                intervention: An instance of
                    interfere.interventions.ExogIntervention.
        
        Note:
            Args are passed via decorators for the test class.
        """

        (prior_endog_states, prior_exog_states, prior_t, endo_true, 
         exog, forecast_times, intervention) = dynamics
        
        method_params = method_type.get_test_params()
        method = method_type(**method_params)

        method.fit(prior_t, prior_endog_states, prior_exog_states)

        endo_pred = method.predict(
            forecast_times,
            prior_endog_states,
            prior_exog_states,
            prior_t=prior_t,
            prediction_exog=exog,
        )

        assert endo_pred.shape[1] == endo_true.shape[1]
        assert endo_pred.shape[0] == PRED_LEN


    def test_predict_clip(
            self, method_type: Type[interfere.ForecastMethod], dynamics):
        """Tests that the predict method clips predictions when they get big.
        
        Args:
            method_type: An interfere.ForecastMethod
                method type.
            dynamics: A tuple containing
                prior_endog_states: An array of historic endogenous states.
                prior_exog_states: An array of historic exogenous states.
                prior_t: Time points corresponing to the rows of the two arrays
                    above. 
                endo_true: An array of endog states for forecasting.
                exog: An array of exog states to be used in forecasting.
                forecast_times: Times corresponidng to the rows of the two
                    arrays above.
                intervention: An instance of
                    interfere.interventions.ExogIntervention.
        
        Note:
            Args are passed via decorators for the test class.
        """

        (prior_endog_states, prior_exog_states, prior_t, endo_true, 
         exog, forecast_times, intervention) = dynamics
        
        method_params = method_type.get_test_params()
        method = method_type(**method_params)

        method.fit(prior_t, prior_endog_states, prior_exog_states)

        prediction_max = max(
            np.max(prior_endog_states), np.max(prior_exog_states))
        
        endo_pred = method.predict(
            t=forecast_times,
            prior_endog_states=prior_endog_states,
            prior_exog_states=prior_exog_states,
            prior_t=prior_t,
            prediction_exog=exog,
            prediction_max=prediction_max
        )
        
        assert np.all(endo_pred <= prediction_max)


    def test_predict_arbitrary_initial(
            self, method_type: Type[interfere.ForecastMethod], dynamics):
        """Tests that predict works for arbitrary initial conditions instead of
        just from the fitted data.
        
        Args:
            method_type: An interfere.ForecastMethod
                method type.
            dynamics: A tuple containing
                prior_endog_states: An array of historic endogenous states.
                prior_exog_states: An array of historic exogenous states.
                prior_t: Time points corresponing to the rows of the two arrays
                    above. 
                endo_true: An array of endog states for forecasting.
                exog: An array of exog states to be used in forecasting.
                forecast_times: Times corresponidng to the rows of the two
                    arrays above.
                intervention: An instance of
                    interfere.interventions.ExogIntervention.
        
        Note:
            Args are passed via decorators for the test class.
        """
        if method_type is interfere.methods.AverageMethod:
            pytest.skip()

        (prior_endog_states, prior_exog_states, prior_t, endo_true, 
         exog, forecast_times, intervention) = dynamics
        
        method_params = method_type.get_test_params()
        method = method_type(**method_params)

        method.fit(prior_t, prior_endog_states)

        # Make random initial conditions.
        rng = np.random.default_rng(SEED)
        random_initial = rng.random(
            (method.get_window_size(), prior_endog_states.shape[1]))

        # Predict from random initial conditions.
        endog_pred = method.predict(
            forecast_times, prior_endog_states=random_initial)
        assert np.allclose(random_initial[-1, :], endog_pred[0, :], atol=0.1), (
            "Arbitrary initial prediction failed. \n\nTarget = "
            f" {random_initial[-1, :]} \n"
            f"Predic = {endog_pred[0, :]}"
        )

        # Check that the model can make multiple predictions from different
        # initial conditions. 
        random_initial2 = rng.random(
            (method.get_window_size(), prior_endog_states.shape[1]))
        
        endog_pred2 = method.predict(
            forecast_times, prior_endog_states=random_initial2)
        
        assert np.allclose(
            random_initial2[-1, :], endog_pred2[0, :], atol=0.1), (
            "Second arbitrary prediction failed. \nTarget = "
            f" {random_initial2[-1, :]} \n"
            f"Predic = {endog_pred2[0, :]}"
        )


@pytest.mark.parametrize("method_type", METHODS)
class TestPredictErrors:


    def test_monotoic_err(self, method_type: Type[interfere.ForecastMethod]):
        """Tests predict() raises an error for non-monotonic time points.
        
        Args:
            method_type: An interfere.ForecastMethod
                method type.
        
        Note:
            Args are passed via decorators for the test class.

        """
        method_params = method_type.get_test_params()

        # Warning and exception tests.
        t = np.arange(10)
        prior_endog_states = np.random.rand(9, 2)
        true_prior_t = np.arange(-8, 1)
        method = method_type(**method_params)
        method.fit(true_prior_t, prior_endog_states)

        # Test that predict requires monotonic time arrays.
        with pytest.raises(ValueError, match=(
            f"Time points passed to the {str(type(method).__name__)}.predict "
            "`t` argument must be strictly increasing."
        )):
            method.predict(np.random.rand(10), prior_endog_states)


    def test_infer_prior_t_warn(self, method_type: Type[interfere.ForecastMethod]):
        """Test that predict() warns about inferring prior_t.
        
        Args:
            method_type: An interfere.ForecastMethod
                method type.
        
        Note:
            Args are passed via decorators for the test class.

        """
        method_params = method_type.get_test_params()

        t = np.arange(10)
        prior_endog_states = np.random.rand(9, 2)
        true_prior_t = np.arange(-8, 1)
        method = method_type(**method_params)
        method.fit(true_prior_t, prior_endog_states)

        # Test that predict warns about inferring prior_t.
        with pytest.warns(UserWarning, match=(
            "Inferring additional `prior_t` values. Assuming `prior_t` has"
            " the same equally spaced timestep size as `t`"
        )):
            method.predict(t, prior_endog_states[-1,:], prior_t=t[0:1])


    def test_equally_spaced_warn(self, method_type: Type[interfere.ForecastMethod]):
        """Test that predict requires equally spaced t to infer prior_t.

        Args:
            method_type: An interfere.ForecastMethod
                method type.
        
        Note:
            Args are passed via decorators for the test class.

        """
        method_params = method_type.get_test_params()

        t = np.arange(10)
        prior_endog_states = np.random.rand(9, 2)
        true_prior_t = np.arange(-8, 1)
        method = method_type(**method_params)
        method.fit(true_prior_t, prior_endog_states)

        # Test that predict requires equally spaced t to infer prior_t.
        with pytest.raises(ValueError, match=(
            "The `prior_t` argument not provided"
            " AND `t` is not equally spaced. Cannot infer "
            " `prior_t`. Either pass it explicitly or provide "
            " equally spaced time `t`."
        )):
            method.predict(np.hstack([
                t, [t[-1] + np.pi]]), prior_endog_states[-1,:], prior_t=None)


    def test_last_prior_t_equals_first_t(self, method_type: Type[interfere.ForecastMethod]):
        """Test that predict requires last entry of prior_t to equal first entry
        of t.  

        Args:
            method_type: An interfere.ForecastMethod
                method type.
        
        Note:
            Args are passed via decorators for the test class.

        """
        method_params = method_type.get_test_params()

        t = np.arange(10)
        prior_endog_states = np.random.rand(9, 2)
        true_prior_t = np.arange(-8, 1)
        method = method_type(**method_params)
        method.fit(true_prior_t, prior_endog_states)

        # Test that predict requires last entry of prior_t to equal first entry
        # of t. 
        bad_prior_t = np.random.rand(1)
        with pytest.raises(ValueError, match=re.escape(
            f"For {str(type(method).__name__)}.predict, the last prior time, "
            f"prior_t[-1]={bad_prior_t[-1]} must equal the first simulation "
            f"time t[0]={t[0]}."
        )):
            method.predict(t, prior_endog_states[-1,:], prior_t=bad_prior_t)    
    

    def test_prior_t_match_prior_endog(self, method_type: Type[interfere.ForecastMethod]):
        """Test that predict requires the same number of entries in prior_t and
        prior_endog_states.  

        Args:
            method_type: An interfere.ForecastMethod
                method type.
        
        Note:
            Args are passed via decorators for the test class.

        """
        method_params = method_type.get_test_params()

        t = np.arange(10.0)
        prior_endog_states = np.random.rand(9, 2)
        true_prior_t = np.arange(-8.0, 1.0)
        method = method_type(**method_params)
        method.fit(true_prior_t, prior_endog_states)

        # Test that predict requires the same number of entries in prior_t and
        # prior_endog_states.
        p = 3
        num_prior_times = 2
        bad_prior_t = true_prior_t[-num_prior_times:].copy()
        bad_prior_t[0] -= np.pi

        # For this test to pass, method window size must be smaller than 3.
        method.get_window_size = lambda : 2

        with pytest.raises(ValueError, match=re.escape(
            f"{str(type(method).__name__)}.predict was passed {p} "
            "prior_endog_states but there are only "
            f"{num_prior_times} entries in `prior_t`."
        )):
            method.predict(t, prior_endog_states[-p:,: ], prior_t=bad_prior_t)

        # Test that predict requires monotonic time arrays.
        with pytest.raises(ValueError, match=re.escape(
            f"Prior time points passed to {str(type(method).__name__)}."
            "predict must be strictly increasing."
        )):
            bad_prior_t = np.random.rand(len(true_prior_t))
            bad_prior_t[-1] = t[0]
            method.predict(t, prior_endog_states, prior_t=bad_prior_t)


    def test_window_size_warn(self, method_type: Type[interfere.ForecastMethod]):
        """Test that predict raises warning when not enough historic data is
        passed.

        Args:
            method_type: An interfere.ForecastMethod
                method type.
        
        Note:
            Args are passed via decorators for the test class.
        
        Note: Changes the window_size() method. This is a bit hacky and could
        cause problems in the future. The real issue here is that window size is
        not an attribute and therefore we can't change it. It is computed from
        the internal method parameters which we have no knowledge of at this
        scope. Therefore this hack is used instead of reconfiguring the test API
        and  requring each method to provide parameters that lead to a window
        size greater than 2. Additionally a window size smaller than 2 will
        throw an error and some methods don't have parameters that generate a
        window size bigger than 2. So a hack is where we are at for now.
        """
        method_params = method_type.get_test_params()

        t = np.arange(10)
        prior_endog_states = np.random.rand(9, 2)
        prior_exog_states = np.random.rand(9, 3)
        prediction_exog = np.random.rand(len(t), 3)
        true_prior_t = np.arange(-8, 1)
        method = method_type(**method_params)
        method.fit(true_prior_t, prior_endog_states, prior_exog_states)
    
        # ensure that the warning is raised correctly.
        w_old = method.get_window_size()
        method.get_window_size = lambda : w_old + 1

        with pytest.warns(UserWarning, match=str(type(method).__name__) + " has window size "
            f"{w_old + 1} but only recieved {w_old} "
            "endog observations. Augmenting historic edogenous "
            "observations with zeros."
        ):
            method.predict(
                t,
                prior_endog_states=prior_endog_states[-w_old:, :],
                prior_exog_states=prior_exog_states[-w_old:, :],
                prior_t=true_prior_t[-w_old:], 
                prediction_exog=prediction_exog,
                prediction_max=3.0
            )


    def test_window_size_warn_exog(self, method_type: Type[interfere.ForecastMethod]):
        """Test that predict raises warning when not enough historic data is
        passed.

        Args:
            method_type: An interfere.ForecastMethod
                method type.
        
        Note:
            Args are passed via decorators for the test class.
        
        Note: Changes the window_size() method. This is a bit hacky and could
        cause problems in the future. The real issue here is that window size is
        not an attribute and therefore we can't change it. It is computed from
        the internal method parameters which we have no knowledge of at this
        scope. Therefore this hack is used instead of reconfiguring the test API
        and  requring each method to provide parameters that lead to a window
        size greater than 2. Additionally a window size smaller than 2 will
        throw an error and some methods don't have parameters that generate a
        window size bigger than 2. So a hack is where we are at for now.
        """
        method_params = method_type.get_test_params()

        t = np.arange(10)
        prior_endog_states = np.random.rand(9, 2)
        prior_exog_states = np.random.rand(9, 3)
        prediction_exog = np.random.rand(len(t), 3)
        true_prior_t = np.arange(-8, 1)
        method = method_type(**method_params)
        method.fit(true_prior_t, prior_endog_states, prior_exog_states)
    
        # Change window size function (hack)
        w_old = method.get_window_size()
        method.get_window_size = lambda : w_old + 1

        with pytest.warns(UserWarning, match=str(type(method).__name__) + " has window size"
            f" {w_old + 1} but only recieved {w_old} exog observations. "
            "Augmenting historic exogenous observations with zeros."
        ):
            method.predict(
                t=t,
                prior_endog_states=prior_endog_states[-w_old:, :],
                prior_exog_states=prior_exog_states[-w_old:, :],
                prior_t=true_prior_t[-w_old:],
                prediction_exog=prediction_exog, 
                prediction_max=3.0
            )


    def test_prior_t_infer_warning(self, method_type: Type[interfere.ForecastMethod]):
        """Tests that a warning appears when inferring prior_t
        
        Args:
            method_type: An interfere.ForecastMethod
                method type.
        
        Note:
            Args are passed via decorators for the test class.
        
        Note: Changes the window_size() method. This is a bit hacky and could
        cause problems in the future. The real issue here is that window size is
        not an attribute and therefore we can't change it. It is computed from
        the internal method parameters which we have no knowledge of at this
        scope. Therefore this hack is used instead of reconfiguring the test API
        and  requring each method to provide parameters that lead to a window
        size greater than 2. Additionally a window size smaller than 2 will
        throw an error and some methods don't have parameters that generate a
        window size bigger than 2. So a hack is where we are at for now.
        """
        method_params = method_type.get_test_params()
        method = method_type(**method_params)

        # Change window size function (hack)
        w_old = method.get_window_size()
        method.get_window_size = lambda : w_old + 1

        X = np.random.rand(10, 2)
        t = np.arange(10)
        true_prior_t = np.arange(-9, 1)
        method.fit(true_prior_t, X)

        with pytest.warns(UserWarning, match=(
                "Inferring additional `prior_t` values. Assuming `prior_t` has"
                " the same equally spaced timestep size as `t`"
        )):
            method.predict(t, X[-w_old:], prior_t=true_prior_t[-w_old:])


    def test_prior_t_infer_error(self, method_type: Type[interfere.ForecastMethod]):
        """Tests that an error is thrown when prior_t is not equally spaced.
        
        Args:
            method_type: An interfere.ForecastMethod
                method type.
        
        Note:
            Args are passed via decorators for the test class.
        
        Note: Changes the window_size() method. This is a bit hacky and could
        cause problems in the future. The real issue here is that window size is
        not an attribute and therefore we can't change it. It is computed from
        the internal method parameters which we have no knowledge of at this
        scope. Therefore this hack is used instead of reconfiguring the test API
        and  requring each method to provide parameters that lead to a window
        size greater than 2. Additionally a window size smaller than 2 will
        throw an error and some methods don't have parameters that generate a
        window size bigger than 2. So a hack is where we are at for now.
        """
        method_params = method_type.get_test_params()
        method = method_type(**method_params)

        # Change window size function (hack)
        w_old = method.get_window_size()
        method.get_window_size = lambda : w_old + 1

        X = np.random.rand(10, 2)
        t = np.arange(10)
        true_prior_t = np.arange(-9, 1)
        method.fit(true_prior_t, X)

        # Test that predict requires equally spaced t to infer prior_t.
        with pytest.raises(ValueError, match=re.escape(
            f"{str(type(method).__name__)}.predict augmented "
            "`prior_endog_states` with zeros but `prior_t` was not "
            "equally spaced so it was not possible to infer "
            "additional prior times. \n\nTo solve, pass at least "
            f"({w_old + 1}) previous time values or use "
            "equally spaced `prior_t`."
        )):
            bad_prior_t = true_prior_t[-w_old:].copy()
            bad_prior_t[0] -= np.pi
            method.predict(t, X[-w_old:], prior_t=bad_prior_t)


@pytest.mark.parametrize("method_type", METHODS)
def test_continuous_predict_arbitrary_initial(
    method_type: Type[interfere.ForecastMethod]):
    """Test that the method can predict continuous time series data.
        
        Args:
            method_type: The type of the ForecastMethod.
    """
    if method_type is interfere.methods.AverageMethod:
        pytest.skip()

    dt = 1/100
    method = method_type(**method_type.get_test_params())

    prior_t = np.arange(0, 2 * np.pi, dt)
    prior_endog_states = np.vstack([np.sin(prior_t), np.cos(prior_t)]).T

    method.fit(prior_t, prior_endog_states)

    rng = np.random.default_rng(SEED)
    rand_t0 = rng.random()
    pred_t = np.array([rand_t0, rand_t0 + dt])
    rand_prior_t = np.arange(-20, 1) * dt + rand_t0
    rand_prior_states = np.vstack([
        np.sin(rand_prior_t),
        np.cos(rand_prior_t)
    ]).T
    
    endo_pred = method.predict(
        pred_t, rand_prior_states, prior_t=rand_prior_t)
    
    assert np.allclose(endo_pred[0, :], endo_pred[1, :], atol=dt**0.5), (
        "Change in state too big for continuous system:"
        f"\nTarget={endo_pred[0, :]}"
        f"\nPredic={endo_pred[1, :]}"
    )

    assert not np.allclose(endo_pred[0, :], endo_pred[1, :]), (
        "Change in state not big enough for the modeled system."
        f"\nTarget={endo_pred[0, :]}"
        f"\nPredic={endo_pred[1, :]}"
    )


@pytest.mark.parametrize("method_type", EXOG_RESP_METHODS)
def test_optimize_method_exog_response(
    method_type: Type[interfere.ForecastMethod]):
    """Test that optimizer can get the method to correctly anticipate the
    intervention response.
    
    Args:
        method_type: A subtype of the ForecastMethod.

    Note:
        Args are provided via parametrize decorator.
    """
    (
        train_t, train_states, forecast_t, 
        forecast_states, interv_states, intervention
    ) = OPTUNA_TRAINING_DATA

    params = method_type.test_exog_response_params()
    method = method_type(**params)

    (endo_train_states, 
        exog_train_states) = intervention.split_exog(
            train_states)
    
    exog_pred = intervention.eval_at_times(
        forecast_t)

    # Fit and forecast intervention.
    method.fit(train_t, endo_train_states, exog_train_states)

    interv_endo_pred = method.predict(
        forecast_t,
        prior_endog_states=endo_train_states, prior_exog_states=exog_train_states,
        prior_t=train_t,
        prediction_exog=exog_pred
    )
    interv_pred = intervention.combine_exog(
        interv_endo_pred, exog_pred)


    control_error = interfere.metrics.RootMeanStandardizedSquaredError()(
        train_states,
        forecast_states,
        interv_pred,
        intervention
    )

    interv_error = interfere.metrics.RootMeanStandardizedSquaredError()(
        train_states,
        interv_states,
        interv_pred,
        intervention
    )

    assert control_error < interv_error, (
        "Intervention response prediction was closer to the control states"
        " than to the intervention response states."
        f"\n\nControl vs. predicted interv. response RMSSE: {control_error}"
        "\n\nInterv. response vs. predicted interv. response RMSSE: "
        f"{interv_error}"
    )


def test_sindy_refit():
    """Tests that sindy libraries and differentiation don't carry over state."""
    for diff in interfere._methods.sindy.SINDY_DIFF_LIST:
        for lib in interfere._methods.sindy.SINDY_LIB_LIST:
            method = interfere.methods.SINDy(
                feature_library=lib, differentiation_method=diff)
            n_obs = 10
            t = np.arange(n_obs)
            # Three endog, no exog.
            X = np.random.rand(n_obs, 3)
            method.fit(t, X)
            pred = method.predict(
                np.arange(t[-1], t[-1] + 5),
                X,
                prior_t = t
            )
            assert pred.shape == (5, 3)
            

            # Two endog, two exog.
            X = np.random.rand(n_obs, 2)
            X_ex = np.random.rand(n_obs, 2)
            method.fit(t, X)
            pred_ex = np.random.rand(5, 2)
            pred = method.predict(
                np.arange(t[-1], t[-1] + 5),
                X,
                prior_t = t,
                prediction_exog=pred_ex,
            )
            assert pred.shape == (5, 2)

            # One endog, no exog.
            X = np.random.rand(n_obs, 1)
            method.fit(t, X)
            pred = method.predict(
                np.arange(t[-1], t[-1] + 5),
                X,
                prior_t = t
            )
            assert pred.shape == (5, 1)