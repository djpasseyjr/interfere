""""""""
from typing import Optional, Type, Union
from warnings import warn

import numpy as np
import pysindy as ps


try:
    import tigramite
    from tigramite.pcmci import PCMCI
    from tigramite import data_processing as pp
    from tigramite.independence_tests.cmiknn import CMIknn
    from tigramite.independence_tests.parcorr import ParCorr
except ImportError as err:
    raise ImportError(
        "An import error occurred while importing tigramite: "
        f"\n\n\t{err}.\n\n"
        "Note that Tigramite may not be installed. You can install it"
        " via \n\n\tpip install tigramite\n\nAlternatively, you can "
        "install all method dependencies via\n\tpip install interfere[methods]"
    )

try:
    import surd

except ImportError as err:
    raise ImportError(
        "An import error occurred while importing surd: "
        f"\n\n\t{err}.\n\n"
        "The interfere package uses a special fork of surd because the "
        "original SURD software is not a pypi package. You can install it via:"
        "\n\n\tpip install git+https://github.com/djpasseyjr/surd.git"
    )

from ..base import ForecastMethod, DEFAULT_RANGE
from .._methods.sindy import SINDy, SINDY_DIFF_LIST, SINDY_LIB_LIST
from ..utils import copy_doc


PCMCI_COND_IND_TEST_LIST = [ParCorr, CMIknn]


class PCMCI_SINDy(ForecastMethod):
    """Combines PCMCI and SINDy."""

    def __init__(
        self,
        cond_ind_test: Type[
            tigramite.independence_tests.independence_tests_base.CondIndTest],
        cond_ind_pval: float = 0.01,
        optimizer: Optional[ps.optimizers.BaseOptimizer] = None,
        feature_library: Optional[
            ps.feature_library.base.BaseFeatureLibrary] = None,
        differentiation_method: Optional[
            ps.differentiation.BaseDifferentiation] = None,
        feature_names: Optional[list[str]] = None,
        t_default: int = 1,
        discrete_time: bool = False,
        max_sim_value: int = 10000,
        **kwargs
    ):
        """Run PCMCI and restrict SINDy to use only links supported by PCMCI.

        Args:
            cond_ind_test (CondIndTest): A tigramite conditional independence test.
                For example, can be ParCorr or CMIknn.
            cond_ind_pval (float): Significance level for conditional independence
                test.
            optimizer (BaseOptimizer): Optimization method used to fit the SINDy
                model. Default is `ps.STLSQ()`.
            feature_library (BaseFeatureLibrary): Feature library object used to
                specify candidate right-hand side features.
                The default option is `PolynomialLibrary`.
            differentiation_method (BaseDifferentiation):
                Method for differentiating the data. The default option is centered
                difference.
            feature_names : list of string, length n_input_features, optional
                Names for the input features (e.g. ``['x', 'y', 'z']``). If None, will use
                ``['x0', 'x1', ...]``.
            t_default : float, optional (default 1)
                Default value for the time step.
            discrete_time : boolean, optional (default False)
                If True, dynamical system is treated as a map. Rather than
                predictingderivatives, the right hand side functions step the system
                forward by one time step. If False, dynamical system is assumed to
                be a flow (right-hand side functions predict continuous time
                derivatives).
            max_sim_value : float, optional (default 10000) Absolute max value for
                state during simulation. Prevents simulation from diverging.

        Note: All kwargs are passed to an internal interfere.SINDy class.
        """
        self.cond_ind_test = cond_ind_test()
        self.cond_ind_pval = cond_ind_pval
        self.sindy_params = dict(
            optimizer=optimizer,
            feature_library=feature_library,
            differentiation_method=differentiation_method,
            feature_names=feature_names,
            t_default=t_default,
            discrete_time=discrete_time,
            max_sim_value=max_sim_value,
            **kwargs
        )


    @copy_doc(ForecastMethod._fit)
    def _fit(
        self,
        t: np.ndarray,
        endog_states: np.ndarray,
        exog_states: Optional[np.ndarray] = None
    ):
        # Sample fit of SINDy to get variable and feature names.
        sample_fit = SINDy(**self.sindy_params)

        # If exogenous states are not provided, set them to zero.
        sample_exog = None if exog_states is None else exog_states[:2, :]
        # Fit SINDy to the first two samples of the data.
        sample_fit.fit(
            t[:2],
            endog_states[:2, :],
            sample_exog
        )
        
        self.var_names = sample_fit.sindy.feature_names
        self.feat_names = sample_fit.sindy.feature_library.get_feature_names()
        
        # Build the feature mask.
        feature_mask = self.restrict_features(
            t, endog_states, exog_states, self.var_names, self.feat_names
        )
        # We only need the mask for the endogenous variables.
        feature_mask = feature_mask[:endog_states.shape[1], :]

        # Create optimizer that limits different features for different
        # variables. This is done by using a feature mask.
        opt = sample_fit.sindy.optimizer
        opt_reinitialized = type(opt)(**opt.get_params())
        restricted_optimizer = FeatureMaskOptimizer(
            opt_reinitialized, feature_mask
        )
        self.sindy = SINDy(**{
            **self.sindy_params,
            "optimizer": restricted_optimizer
        })
        self.sindy.fit(t, endog_states, exog_states)


    @copy_doc(ForecastMethod._predict)
    def _predict(
        self,
        t: np.ndarray,
        prior_endog_states: np.ndarray,
        prior_exog_states: Optional[np.ndarray] = None,
        prior_t: Optional[np.ndarray] = None,
        prediction_exog: Optional[np.ndarray] = None,
        rng: np.random.RandomState = DEFAULT_RANGE,
    ) -> np.ndarray:
        return self.sindy.predict(
            t=t,
            prior_endog_states=prior_endog_states,
            prior_exog_states=prior_exog_states,
            prior_t=prior_t,
            prediction_exog=prediction_exog,
            rng=rng
        )
 

    def restrict_features(
        self,
        t: np.ndarray,
        endog_states: np.ndarray,
        exog_states: Optional[np.ndarray],
        var_names: list[str],
        feature_names: list[str]
    ):
        """Uses PCMCI to restrict the set of features used in the SINDy model.

        Args:
            t (np.ndarray): Array with shape (num_samples,).
            endog_states (np.ndarray): 
                Array with shape (num_samples, num_endo_vars).
            exog_states (np.ndarray):
                Array with shape (num_samples, num_exog_vars).
            var_names (list[str]): List of variable names. First `num_endo_vars`
                variables correspond to endogenous and the remaining correspond 
                to exogenous variables. Can only contain 
                letters, numbers and underscores. E.g. x0, x_1....
        feature_names (list[str]): List of feature names. E.g. x0^2. A feature 
            is assumed to contain a substring containing a variable's name **if
            and only if** that variable is used to compute that feature.

        Returns:
            feature_mask (np.ndarray): Feature mask of shape (num_endo_vars, 
                num_features). If feature_mask[i, j] == True, then feature j is 
                allowed in the equation for variable i. Otherwise, it is 
                excluded from the equation.
        """
        # Check if the time series is discrete or continuous.
        if np.any(np.diff(t) != 1):
            warn(
                "Our PCMCI adapter only supports discrete time series. "
                "Please provide a time series with a time step of 1."
                f"Provided time series (first 5 entries): \n\t{t[:5]}"
                "\n\nNote: This restriction exists because our initial testing "
                "found that differential equations have too much memory and "
                "seem to confuse PCMCI."
            )
        
        if exog_states is None:
            exog_states = np.zeros((endog_states.shape[0], 0))

        # Concatenate endogenous and exogenous states.
        dataframe = pp.DataFrame(
            np.hstack((endog_states, exog_states)),
            var_names=self.var_names
        )

        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=self.cond_ind_test,
            verbosity=0
        )

        results = pcmci.run_pcmciplus(
            tau_min=0,
            tau_max=1,
            pc_alpha=self.cond_ind_pval
        )

        # Transform causal graph from PCMCI to a feature mask.
        pcmci_lag1_var_adj = pcmci_graph_to_adjacency_matrix(
            results['graph'], lag=1)
        feature_mask = variable_adjacency_matrix_to_feature_mask(
            var_names, feature_names, pcmci_lag1_var_adj
        )
        return feature_mask
    
    @staticmethod
    @copy_doc(ForecastMethod._get_optuna_params)
    def _get_optuna_params(trial, **kwargs):
        return {
            'cond_ind_test': trial.suggest_categorical(
                'cond_ind_test', PCMCI_COND_IND_TEST_LIST),

            'cond_ind_pval': trial.suggest_float(
                'cond_ind_pval', 1e-5, 0.1, log=True),

            'optimizer__threshold': trial.suggest_float(
                'optimizer__threshold', 1e-5, 5, log=True),

            'optimizer__alpha': trial.suggest_float(
                'optimizer__alpha', 1e-5, 5, log=True),

            'discrete_time': trial.suggest_categorical(
                'discrete_time', [True, False]),

            'feature_library':
                trial.suggest_categorical('feature_library', SINDY_LIB_LIST),

            'differentiation_method':
                trial.suggest_categorical(
                    "differentiation_method", SINDY_DIFF_LIST)
        }
    

    @staticmethod
    def get_test_params():
        return {
            "cond_ind_test": ParCorr,
            "cond_ind_pval": 0.01,
            **SINDy.get_test_params()
        }
    

class SURD_SINDy(PCMCI_SINDy):
    """Uses SURD to restrict feature selection in SINDy."""

    def __init__(
        self,
        surd_threshold: float = 0.01,
        optimizer: Optional[ps.optimizers.BaseOptimizer] = None,
        feature_library: Optional[
            ps.feature_library.base.BaseFeatureLibrary] = None,
        differentiation_method: Optional[
            ps.differentiation.BaseDifferentiation] = None,
        feature_names: Optional[list[str]] = None,
        t_default: int = 1,
        discrete_time: bool = False,
        max_sim_value: int = 10000,
        **kwargs
    ):
        """Run SURD and restrict SINDy to use only links supported by SURD.

        Args:
            optimizer (BaseOptimizer): Optimization method used to fit the SINDy
                model. Default is `ps.STLSQ()`.
            feature_library (BaseFeatureLibrary): Feature library object used to
                specify candidate right-hand side features.
                The default option is `PolynomialLibrary`.
            differentiation_method (BaseDifferentiation):
                Method for differentiating the data. The default option is 
                centered difference.
            feature_names : list of string, length n_input_features, optional
                Names for the input features (e.g. ``['x', 'y', 'z']``). If 
                None, will use``['x0', 'x1', ...]``.
            t_default : float, optional (default 1)
                Default value for the time step.
            discrete_time : boolean, optional (default False)
                If True, dynamical system is treated as a map. Rather than
                predictingderivatives, the right hand side functions step the 
                system forward by one time step. If False, dynamical system is 
                assumed to be a flow (right-hand side functions predict 
                continuous time derivatives).
            max_sim_value : float, optional (default 10000) Absolute max value 
                for state during simulation. Prevents simulation from diverging.

        Note: All kwargs are passed to an internal interfere.SINDy class.
        """
        self.surd_threshold = surd_threshold
        self.sindy_params = dict(
            optimizer=optimizer,
            feature_library=feature_library,
            differentiation_method=differentiation_method,
            feature_names=feature_names,
            t_default=t_default,
            discrete_time=discrete_time,
            max_sim_value=max_sim_value,
            **kwargs
        ) 

    def restrict_features(
        self, t, endog_states, exog_states, var_names, feature_names):
        """Uses SURD to restrict the set of features used in the SINDy model.

        Args:
            t (np.ndarray): Array with shape (num_samples,).
            endog_states (np.ndarray):
                Array with shape (num_samples, num_endo_vars).
            exog_states (np.ndarray):
                Array with shape (num_samples, num_exog_vars).
            var_names (list[str]): List of variable names. First `num_endo_vars`
                variables correspond to endogenous and the remaining correspond
                to exogenous variables. Can only contain
                letters, numbers and underscores. E.g. x0, x_1....
            feature_names (list[str]): List of feature names. E.g. x0^2. A 
                feature is assumed to contain a substring containing a 
                variable's name **if and only if** that variable is used to 
                compute that feature.

        Returns:
            feature_mask (np.ndarray): Feature mask of shape (num_endo_vars,
                num_features). If feature_mask[i, j] == True, then feature j is
                allowed in the equation for variable i. Otherwise, it is
                excluded from the equation.
        """
        # Check if the time series is discrete or continuous.
        if np.any(np.diff(t) != 1):
            warn(
                "The interfere SURD adapter only supports discrete time "
                "series. Please provide a time series with a time step of 1."
                f"Provided time series (first 5 entries): \n\t{t[:5]}"
                "\n\nNote: This restriction exists because our initial testing "
                "found that differential equations have too much memory and "
                "seem to confuse SURD."
            )
        if exog_states is None:
            exog_states = np.zeros((endog_states.shape[0], 0))

        # Concatenate endogenous and exogenous states.
        states = np.hstack((endog_states, exog_states))
        num_vars = states.shape[1]

        Rd_results, Sy_results, _, _ = surd.surd_parallel(
            states,
            states,
            nlag=1,
            nbins=15,
            max_combs=2,
            cores=4,
        )

        adj = np.zeros((num_vars, num_vars))

        for i in range(num_vars):

            impactful_vars = set()

            # Compute total information
            redun_total = sum([v for v in Rd_results[i+1].values()])
            syn_total = sum([v for v in Sy_results[i+1].values()])
            info_total = redun_total + syn_total

            # Find all groups whose specific information is greater than the threshold.

            for info_type in [Rd_results[i+1], Sy_results[i+1]]:
                for k, v in info_type.items():
                    if v / info_total > self.surd_threshold:
                        impactful_vars.update(k)

            for j in impactful_vars:
                adj[i, j-1] = True

        # Transform causal graph from PCMCI to a feature mask.
        feature_mask = variable_adjacency_matrix_to_feature_mask(
            var_names, feature_names, adj
        )
        return feature_mask
    

    @staticmethod
    def _get_optuna_params(trial, **kwargs):
        return {
            'surd_threshold': trial.suggest_float(
                'surd_threshold', 0.0, 1.0),
            **SINDy._get_optuna_params(trial, **kwargs)
        }
    

    @staticmethod
    def get_test_params():
        return {
            "surd_threshold": 0.01,
            **SINDy.get_test_params()
        }
    

def pcmci_graph_to_adjacency_matrix(
    pcmci_graph: np.ndarray,
    lag: int = 1
):
    """Transforms a pcmci graph to an adjacency matrix.

    A simple transformation that only uses one lag and does not consider
    contemporaneous causal links. Transposes the pcmci_graph.

    Args:
        pcmci_graph (np.ndarray): Graph of shape (num_vars, num_vars, num_lags)
        lags (int): Number of lags to use.

    Returns:
        adj (np.ndarray): Adjacency matrix of shape (num_vars, num_vars). 
            If adj[i, j] == True, then var_j has a causal influence on var_i.
    """
    if lag > pcmci_graph.shape[2] - 1:
        raise ValueError(
            f"Provided `lag` not compatible with {pcmci_graph.shape=}"
        )

    m, n = pcmci_graph.shape[:2]
    g = pcmci_graph[:, :, lag].T
    adj = np.zeros((m, n), bool)

    for i in range(m):
        for j in range(n):
            if g[i, j] == '-->':
                adj[i, j] = True

            elif g[i, j] == '':
                pass
            
            else:
                raise ValueError(
                    "Invalid edge type: "
                    f"pcmci_graph[{i}, {j}, {lag}] = {pcmci_graph[i, j, lag]}"
                )

    return adj

def variable_adjacency_matrix_to_feature_mask(
    variable_names: list[str],
    feature_names: list[str],
    adjacency_matrix: np.ndarray,
):
    """Convert a variable adjacency matrix to a feature mask.

    Features are allowed by default, so if a feature uses no variables then
    it will be left marked True in the feature mask.

    Args:
        variable_names (list[str]): List of variable names. Can only contain 
        letters, numbers and underscores. E.g. x0, x_1....
        feature_names (list[str]): List of feature names. E.g. x0^2. A feature 
            is assumed to contain a substring containing a variable's name **if
            and only if** that variable is used to compute that feature.
        adjacency_matrix (np.ndarray): Adjacency matrix of shape
            (num_variables, num_variables). If adjacency_matrix[i, j] == True,
            then var_j is allowed in the equation for var_i. Otherwise, it is
            excluded.

    Returns:
        feature_mask (np.ndarray): Feature mask of shape (num_variables, 
            num_features). If feature_mask[i, j] == True, then feature j is 
            allowed in the equation for variable i. Otherwise, it is excluded.
    """
    # Check that each variable name contains only letters, numbers an underscores.
    for var_name in variable_names:
        for char in var_name:
            if not char.isalnum() and char != "_":
                raise ValueError(
                    "Variable names must contain only letters, numbers, and"
                    f" underscores. Incorrect variable name: {var_name}"
                )


    num_variables = len(variable_names)
    num_features = len(feature_names)
    feature_mask = np.ones((num_variables, num_features), dtype=bool)

    # For each equation...
    for i in range(num_variables):
        # For each variable...
        for j in range(num_variables):

            # If the variable is not allowed to appear in the equation
            if not adjacency_matrix[i, j]:

                # Mark every feature containing that variable as False.
                for k in range(num_features):
                    if variable_names[j] in feature_names[k]:
                        feature_mask[i, k] = False

    return feature_mask


class FeatureMaskOptimizer(ps.BaseOptimizer):
    def __init__(
        self,
        optimizer: ps.BaseOptimizer,
        feature_mask: np.ndarray,
    ):
        """Initialize the FeatureMaskOptimizer class.

        Args
            optimizer: SINDy optimizer object used to solve coefficent estimation.
            feature_mask (np.ndarray): Mask array used to select features. Mask array shape
                must be (num_vars, num_features_in_lib). Ones/True denote inclusion
                and zeros/False denote exclusion. For example
                feature_mask[i, j] == 0 means that feature j is not allowed in the
                equation for variable i.
            

        Raises:
            ValueError: If feature mask contains an entry besides zero and one.
        """
        # Check that feature mask contains only zeros and ones.
        if not np.all((feature_mask == 0) | (feature_mask == 1)):
            raise ValueError("Feature mask must contain only zeros and ones.")

        self.optimizer = optimizer
        self.feature_mask = feature_mask.astype(bool)
        super().__init__()


    def mask_features(
        self, x: Union[np.ndarray, ps.AxesArray],
        target_idx: int
    ):
        """Performs masking of the feature matrix.

        Args:
            x (np.ndarray): Array with shape (num_samples, num_features).
                Contains all features from the feature library.
            target_idx (int): Index of the target target derivative in
                self.feature_mask.

        Returns:
            np.ndarray: Masked feature matrix. Shape is will be
                (num_samples, num_non_masked_features). Here,
                `num_non_masked_features = sum(self.feature_mask[i, :])`.
        """
        mask = self.feature_mask[target_idx, :]
        masked_x = x[:, mask]
        return masked_x


    def mask_and_opt(
        self,
        x: Union[np.ndarray, ps.AxesArray],
        y: Union[np.ndarray, ps.AxesArray],
    ):
        """Solves for system coeffients by looping through targets, masking
        features and using the internal optimizer to compute coeffients for each
        target.

        Args:
            x (np.ndarray): Array with shape (num_samples, num_features).
                Contains all features from the feature library.
            y (np.ndarray): Array with shape (num_samples, num_targets).

        Returns:
            None

        Raises:
            ValueError: If x and y do not have the correct shape.
        """
        if (len(x.shape) != 2) or (len(y.shape) != 2):
            raise ValueError("x and y must be 2D arrays.")

        num_features = x.shape[1]
        # Validate initialization arguments.
        if self.feature_mask.shape[1] != num_features:
            raise ValueError(
                "Passed feature array shape does not match feature_mask shape."
                f"\n\tFeature array shape: (num_samples = {x.shape[0]}, "
                f"num_features = {x.shape[1]})."
                f"\n\tFeature mask shape: (num_targets = {self.feature_mask.shape[0]},"
                f" num_features = {self.feature_mask.shape[1]})"
            )

        num_targets = y.shape[1]
        if num_targets != self.feature_mask.shape[0]:
            raise ValueError(
                "Passed target array shape does not match feature_mask shape."
                f"\n\tTarget array shape: (num_samples = {y.shape[0]}, "
                f"num_targets = {y.shape[1]})."
                f"\n\tFeature mask shape: (num_targets = {self.feature_mask.shape[0]},"
                f" num_features = {self.feature_mask.shape[1]})"
            )

        full_coefs = np.zeros((num_targets, num_features))

        for target_idx in range(num_targets):
            # Access feature mask for target.
            mask = self.feature_mask[target_idx, :]

            # Remove features that should not affect target i.
            masked_x = self.mask_features(x, target_idx)

            # Get target i.
            target_y = y[:, target_idx:target_idx + 1]

            # Check if any features are not masked. (Leaves full_coefs as zeros 
            # if all features are masked.)
            if np.any(mask):
                # Solve optimization for target i.
                self.optimizer.fit(masked_x, target_y)

                # Assign coefs to full coef matrix
                full_coefs[target_idx, mask] = self.optimizer.coef_

        return full_coefs


    def _reduce(
        self,
        x: Union[np.ndarray, ps.AxesArray],
        y: Union[np.ndarray, ps.AxesArray],
        **kwargs
    ):
        """Implements abstract function for pysindy.BaseOptimizer._reduce

        Mutates self.coefs_

        Args:
            x (np.ndarray): Array with shape (num_samples, num_features).
                Contains all features from the feature library.
            y (np.ndarray): Array with shape (num_samples, num_targets).

        Returns:
            None
        """
        coef = self.mask_and_opt(x, y)
        self.coef_ = coef


    def get_params(self, deep = True):
        return {
            **self.optimizer.get_params(deep=deep), 
            "feature_mask": self.feature_mask
        }
    
    def set_params(self, **params):
        # Set params for the internal optimizer.
        self.feature_mask = params.pop("feature_mask", self.feature_mask)
        self.optimizer.set_params(**params)
        return self