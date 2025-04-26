from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import traceback

import interfere
from interfere.metrics import rmse
import numpy as np
from optuna.trial import Trial


class CrossValObjective:
    """Cross validation objective for forecasting/intervention prediction.
    
    Compatible with optuna.
    """


    def __init__(
        self,
        method_type: Type[interfere.ForecastMethod],
        data: np.ndarray,
        times: np.ndarray,
        train_window_percent: float,
        num_folds: int,
        exog_idxs: Optional[list[int]] = None,
        val_scheme: str = "forecast",
        num_val_prior_states: int = 10,
        metric: Union[
            interfere.metrics.CounterfactualForecastingMetric,
            Callable[[np.ndarray, np.ndarray], float]
         ] = rmse,
        metric_direction: str = "minimize",
        hyperparam_func: Optional[
            Callable[[Trial], Dict[str, Any]]] = None,
        store_preds: bool = True,
        raise_errors: bool = False,
        repl_nan_val: float = 1e10,
    ):
        """Initializes a cross validation scheme for forecasting or intervention prediction.

        This creates an objective function that is compatible with optuna:
        
        ```
        cvo = CrossValObjective(*args, **kwargs)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective=cvo, n_trials=100)
        ```

        Args:
            method_type (Type[interfere.ForecastMethod]): A
                forecasting method from interfere.
            data (np.ndarray): A two dimensional array of time series
                states. Columns are variables and rows are observations
                corresponding to `times`.
            times (np.ndarray): A one dimensional array of times
                corresponding to the rows of data.
            train_window_percent (float): The percent of data to use in the
                sliding training window.
            num_folds (int): The number of times to slide the training window
                forward. (Amount of training data stays fixed.)
            val_scheme (str): The type of validation scheme to use.
                One of ["forecast", "last", "all"]. See notes section for
                more details.
            exog_idxs (list[int]): A list of exogenous variable column indexes.
            num_val_prior_states (int): Designates how many observations to use
                as initial condition/prior state for prediction.
            metric (interfere.metrics.CounterfactualForecastingMetric or
                callable[[np.ndarray, np.ndarray], float]): Metric to optimize.
                Exogenous variables are excluded from error/accuracy calculation.
            metric_direction (str): Direction to optimize. One of ["maximize",
                "minimize"].
            hyperparam_func (callable): Accepts an optuna Trial object and
                returns a dictionary of parameters. Defaults to the hyper
                parameter function built into interfere methods.
            store_preds (bool): Toggles if hyper parameter opt predictions
                should be stored. If True, predictions are accessible in self.
                trial_results.
            raise_errors (bool): Toggles if errors should be raised.


        Notes:

            The cross validation schemes works as follows. First, the data
            is divided into a sliding training set and a sliding validation set.

            |-----Train Data-----|-----Val Data-----|

            The validation set is broken into chunks and the "number of folds"
            is equal to the number of validation chunks plus one (for the
            training data fold). For four folds, it looks like this:

            |-----Train Data-----|Val Chunk|Val Chunk|Val Chunk|

            During cross validation, the training window slides forward and
            encompasses the next chunk.

            (1) |-----Train Data-----|Val Chunk|Val Chunk|Val Chunk|
            (2) |Val Chunk|-----Train Data-----|Val Chunk|Val Chunk|
            (3) |Val Chunk|Val Chunk|-----Train Data-----|Val Chunk|
            (4) |Val Chunk|Val Chunk|Val Chunk|-----Train Data-----|


            The chunk(s) used to score the training data depends on the "validation
            scheme". The three validation schemes are

            * `val_scheme == "forecast"` corresponds to standard CV for
            forecasting where the forecaster is scored on the data immediately
            following the training data:

                (1) |-----Train Data-----|SCORE CHUNK|------|------|
                (2) |------|-----Train Data-----|SCORE CHUNK|------|
                (3) |------|------|-----Train Data-----|SCORE CHUNK|
                (4) SKIP (Because there is no chunk after the training data.)

            * `val_scheme == "last"` tells the cross validator to always score
            the forecaster on the last chunk and never allow it to see the data
            that occurred directly before the score chunk.

                (1) |-----Train Data-----|------|------|SCORE CHUNK|
                (2) |------|-----Train Data-----|------|SCORE CHUNK|
                (3) SKIP (Because the last chunk occurs directly after the
                    training data.)
                (4) SKIP (Because there is no chunk after the training data.)

            * `val_scheme == "all"` tells the cross validator to score
            the forecaster on all chunks.

                (1) |-----Train Data-----|SCORE CHUNK|SCORE CHUNK|SCORE CHUNK|
                (2) |SCORE CHUNK|-----Train Data-----|SCORE CHUNK|SCORE CHUNK|
                (3) |SCORE CHUNK|SCORE CHUNK|-----Train Data-----|SCORE CHUNK|
                (4) |SCORE CHUNK|SCORE CHUNK|SCORE CHUNK|-----Train Data-----|
        """
        self.method_type = method_type
        self.data = data
        self.times = times
        self.train_window_percent = train_window_percent
        self.num_folds = num_folds
        self.val_scheme = val_scheme
        self.num_val_prior_states = num_val_prior_states
        self.metric = metric
        self.metric_direction = metric_direction
        # Assign default hyper param func.
        if hyperparam_func is None:
            hyperparam_func = method_type._get_optuna_params 
        self.hyperparam_func = hyperparam_func
        self.store_preds = store_preds
        self.raise_errors = raise_errors
        self.repl_nan_val = repl_nan_val

        # Assign default intervention.
        if exog_idxs == None:
            exog_idxs = []

        # Create a dummy intervention to handle exogenous.
        self.intervention = interfere.PerfectIntervention(
                exog_idxs, [np.nan for _ in exog_idxs])

        self.trial_results = {}

        if self.num_folds <= 1:
            raise ValueError(
                "The number of folds must be greater than 1.\n\n"
            )

        self.num_obs = data.shape[0]
        self.num_val_chunks = num_folds - 1
        self.num_train_obs = int(train_window_percent * self.num_obs)


        self.num_val_chunk_obs = int(
            (self.num_obs - self.num_train_obs) / self.num_val_chunks)

        cv_descr = (
            f"Control V.S. Response Cross Validation Objective"
            f"\n\tValidation scheme: {val_scheme}"
            f"\n\tNumber of folds: {num_folds}"
            f"\n\tNumber of training observations: {self.num_train_obs}"
            f"\n\tNumber of validation chunks: {self.num_val_chunks}"
            f"\n\tNumber of observations per validation chunk: {self.num_val_chunk_obs}"
            f"\n\tNumber of validation prior states: {self.num_val_prior_states}"
        )
        self.cv_descr = cv_descr

        if self.num_val_chunk_obs <= 2:
            raise ValueError(
                "The number of observations in each validation chunk must be "
                "greater than 2.\n\n" + cv_descr
            )

        if self.num_train_obs <= 2:
            raise ValueError(
                "The number of observations in the training set must be "
                "greater than 2.\n\n" + cv_descr
            )

        if val_scheme not in ["forecast", "last", "all"]:
            raise ValueError(
                "The validation scheme must be one of ['forecast', 'last', "
                "'all'].\n\n"
            )

        if self.metric_direction not in ["maximize", "minimize"]:
            raise ValueError(
                "The metric direction must be one of ['maximize', 'minimize'].\n\n"
            )

        if (val_scheme == "forecast") and (
            self.num_train_obs < self.num_val_prior_states):
            raise ValueError(
                "The number of training observations must be greater than "
                "or equal the number of validation prior states when "
                "val_scheme=='forecast'\n\n" + cv_descr
            )

        if (val_scheme in ["last", "all"]) and (
            self.num_val_chunk_obs - self.num_val_prior_states < 2):
            raise ValueError(
                "The number of observations in each validation chunk must be "
                "greater than the number of validation prior states when "
                "val_scheme is 'last' or 'all'.\n\n" + cv_descr
            )

        # Make training window indexes.
        self.train_window_idxs = self._make_train_window_idxs(
            num_train_obs=self.num_train_obs,
            num_val_chunk_obs=self.num_val_chunk_obs,
            num_val_chunks=self.num_val_chunks
        )

        # Make validation chunk indexes.
        self.val_chunk_idxs = self._make_val_chunk_idxs(
            num_train_obs=self.num_train_obs,
            num_val_chunk_obs=self.num_val_chunk_obs,
            num_val_chunks=self.num_val_chunks,
            val_scheme=self.val_scheme
        )


    def __call__(self, trial: Trial) -> float:
        """Run control v.s. response cross validation.

        Args:
            trial: An optuna trial object.

        Returns:
            cv_score: The cross validation score.
        """

        # Initialize method.
        method = self.method_type(
            **self.hyperparam_func(
                trial,
                # Some methods throw errors when the dataset is too small to 
                # accomodate the input size/lags and the forecast horizon.
                max_lags=self.num_val_prior_states,
                max_horizon=self.num_train_obs - self.num_val_prior_states
            ))

        if method.get_window_size() > self.num_val_prior_states:
            raise ValueError(
                "The window size of the method must be less than or equal to "
                "the number of validation prior states.\n\n"
                f"Method: {type(method).__name__}"
                f"\n\tWindow size: {method.get_window_size()}"
                f"\n\tNumber of validation prior states: {self.num_val_prior_states}"
            )

        cv_results = {
            "train_idxs": [],
            "val_prior_idxs": [],
            "val_idxs": [],
            "targets": [],
            "preds": [],
            "scores": [],
            "errors": [],
        }

        if self.store_preds:
            cv_results["data"] = self.data
            cv_results["times"] = self.times
            cv_results["intervention"] = self.intervention
            cv_results["val_scheme"] =  self.val_scheme

        for train_idx, val_idxs in zip(self.train_window_idxs, self.val_chunk_idxs):

            # Check if training fold corresponds to a validation index.
            if val_idxs != []:
                # Prepare train data.
                window_start, window_end = train_idx
                train_t = self.times[window_start:window_end]
                train_states = self.data[window_start:window_end, :]

                train_endog, train_exog = self.intervention.split_exog(
                    train_states)

                # Try except block for method fitting.
                try:
                    method.fit(train_t, train_endog, train_exog)
                    is_fit = True
                    fit_error = "No Fit Error"

                except Exception as e:
                    if self.raise_errors:
                            raise e
                    is_fit = False
                    fit_error = str(e) + f"\n\n{traceback.format_exc()}"
                    


            # Compute validation scores.
            for val_idx in val_idxs:
                if val_idx != []:
                    val_chunk_start, val_chunk_end = val_idx

                    # Collect validation prior states.
                    val_prior_start, val_prior_end = self._make_val_prior_state_idxs(
                        val_chunk_start, self.num_val_prior_states, self.val_scheme
                    )
                    val_prior_t = self.times[val_prior_start:val_prior_end]
                    val_prior_states = self.data[
                        val_prior_start:val_prior_end, :]


                    val_prior_en, val_prior_ex = self.intervention.split_exog(
                        val_prior_states)

                    # Collect validation states.
                    val_start, val_end = self._make_val_idxs(
                        val_chunk_start,
                        val_chunk_end,
                        self.num_val_prior_states,
                        self.val_scheme
                    )

                    val_t = self.times[val_start:val_end]
                    val_states = self.data[val_start:val_end, :]

                    true_val_en, val_ex = self.intervention.split_exog(
                        val_states)

                    try:

                        if not is_fit:
                            raise ValueError(
                                f"Error in model fit: \n\n{fit_error}")

                        # Make prediction.
                        pred_val_endog = method.predict(
                            val_t,
                            prior_endog_states=val_prior_en,
                            prior_exog_states=val_prior_ex,
                            prior_t=val_prior_t,
                            prediction_exog=val_ex
                        )

                        pred_val_states = self.intervention.combine_exog(
                            pred_val_endog, val_ex)

                        # Replace nans.
                        pred_val_states[np.isnan(pred_val_states)] = self.repl_nan_val

                        # Compute score.
                        if isinstance(self.metric, interfere.metrics.CounterfactualForecastingMetric):
                            val_score = self.metric(
                                train_states,
                                val_states,
                                pred_val_states,
                                self.intervention.iv_idxs
                            )
                        else:
                            # Remove exogenous variables from error/accuracy
                            # calculation.
                            val_endog, val_exog = self.intervention.split_exog(
                                val_states)

                            val_score = self.metric(
                                val_endog,
                                pred_val_endog
                            )

                        # Store results.
                        cv_results["scores"].append(val_score)
                        cv_results["errors"].append(None)

                        # Optionally store prediction.
                        if self.store_preds:
                            cv_results["preds"].append(pred_val_states)
                            cv_results["targets"].append(val_states)


                    except Exception as e:

                        if self.raise_errors:
                            raise e

                        # Otherwise, make error log.
                        error_log = str(e) + "\n\n" + traceback.format_exc()
                        # Store error log.
                        cv_results["errors"].append(error_log)
                        cv_results["scores"].append(np.nan)

                        # Optionally store empty predictions.
                        if self.store_preds:
                            cv_results["preds"].append(None)
                            cv_results["targets"].append(val_states)


                    cv_results["train_idxs"].append(train_idx)
                    cv_results["val_prior_idxs"].append((val_prior_start, val_prior_end))
                    cv_results["val_idxs"].append((val_start, val_end))

        # Save results
        self.trial_results[trial.number] = cv_results

        # Return average error.
        return np.mean(cv_results["scores"])

    def _make_train_window_idxs(
        self,
        num_train_obs: int,
        num_val_chunk_obs: int,
        num_val_chunks: int
    ) -> List[Tuple[int, int]]:
        """Creates a list of indexes for each successive training window.

        Args:
            num_train_obs: The number of observations in the training set.
            num_val_chunk_obs: The number of observations in each chunk.
            num_val_chunks: The number of validation chunks.

        Returns:
            train_window_idx: A list of indexes for each successive training window.
        """
        train_window_idxs = [
            (i * num_val_chunk_obs, num_train_obs + i * num_val_chunk_obs)
            for i in range(num_val_chunks + 1)
        ]

        return train_window_idxs


    def _make_val_chunk_idxs(
        self,
        num_train_obs: int,
        num_val_chunk_obs: int,
        num_val_chunks: int,
        val_scheme: str,
    ) -> List[List[Tuple[int, int]]]:
        """Creates a list of indexes for each successive validation window.

        Args:
            num_train_obs: The number of observations in the training set.
            num_val_chunk_obs: The number of observations in each chunk.
            num_val_chunks: The number of validation chunks.
            val_scheme: The validation scheme to use. One of ["forecast",
                "last", "all]. See notes section for more details.

        Returns:
            val_chunk_idxs: A list of lists of indexes for each successive
                validation chunk.

        Notes:

            The cross validation schemes works as follows. First, the data
            is divided into a sliding training set and a sliding validation set.

            |-----Train Data-----|-----Val Data-----|

            The validation set is broken into chunks and the "number of folds"
            is equal to the number of validation chunks plus one (for the
            training data fold). For four folds, it looks like this:

            |-----Train Data-----|Val Chunk|Val Chunk|Val Chunk|

            During cross validation, the training window slides forward and
            encompasses the next chunk.

            (1) |-----Train Data-----|Val Chunk|Val Chunk|Val Chunk|
            (2) |Val Chunk|-----Train Data-----|Val Chunk|Val Chunk|
            (3) |Val Chunk|Val Chunk|-----Train Data-----|Val Chunk|
            (4) |Val Chunk|Val Chunk|Val Chunk|-----Train Data-----|


            The chunk used to score the training data depends on the "validation
            scheme". The three validation schemes are

            * `val_scheme == "forecast"` corresponds to standard CV for
            forecasting where the forecaster is scored on the data immediately
            following the training data:

                (1) |-----Train Data-----|SCORE CHUNK|------|------|
                (2) |------|-----Train Data-----|SCORE CHUNK|------|
                (3) |------|------|-----Train Data-----|SCORE CHUNK|
                (4) SKIP (Because there is no chunk after the training data.)

            * `val_scheme == "last"` tells the cross validator to always score
            the forecaster on the last chunk and never allow it to see the data
            that occurred directly before the score chunk.

                (1) |-----Train Data-----|------|------|SCORE CHUNK|
                (2) |------|-----Train Data-----|------|SCORE CHUNK|
                (3) SKIP (Because the last chunk occurs directly after the
                    training data.)
                (4) SKIP (Because there is no chunk after the training data.)

            * `val_scheme == "all"` tells the cross validator to score
            the forecaster on all chunks.

                (1) |-----Train Data-----|SCORE CHUNK|SCORE CHUNK|SCORE CHUNK|
                (2) |SCORE CHUNK|-----Train Data-----|SCORE CHUNK|SCORE CHUNK|
                (3) |SCORE CHUNK|SCORE CHUNK|-----Train Data-----|SCORE CHUNK|
                (4) |SCORE CHUNK|SCORE CHUNK|SCORE CHUNK|-----Train Data-----|
        """
        if val_scheme == "forecast":
            val_chunk_idxs = [
                [(
                    num_train_obs + i * num_val_chunk_obs,
                    num_train_obs + (i + 1) * num_val_chunk_obs
                )]
                for i in range(num_val_chunks)
            ]

            # Add empty lists for times when there is no validation set that
            # corresponds with the training set.
            val_chunk_idxs += [[]]

        elif val_scheme == "last":
            if num_val_chunks <= 1:
                raise ValueError(
                    "The number of validation chunks must be greater than 1 "
                    "when val_scheme == 'last'."
                )

            val_chunk_idxs = [
                [(
                    num_train_obs + (num_val_chunks - 1) * num_val_chunk_obs,
                    num_train_obs + num_val_chunks * num_val_chunk_obs
                )]
                for i in range(num_val_chunks - 1)
            ]
            # Add empty lists for times when there is no validation set that
            # corresponds with the training set.
            val_chunk_idxs += [[], []]

        elif val_scheme == "all":
            val_chunk_idxs = [
                [
                    # Collect all chunks that occur before training set.
                    (j * num_val_chunk_obs, (j + 1) * num_val_chunk_obs)
                    for j in range(train_set_idx)
                ] + [
                    # Collect all chunks that occur after training set.
                    (num_train_obs + i * num_val_chunk_obs, num_train_obs + (i + 1) * num_val_chunk_obs)
                    for i in range(train_set_idx, num_val_chunks)
                ]
                for train_set_idx in range(num_val_chunks + 1)
            ]

        else:
            raise ValueError(f"Unknown validation scheme: {val_scheme}")

        return val_chunk_idxs


    def _make_val_prior_state_idxs(
        self,
        val_chunk_start: int,
        num_val_prior_states: int,
        val_scheme: str,
    ) -> Tuple[int, int]:
        """Build the indexes for the validation prior states.

        When the val_scheme is forecast, it takes validation prior states from the
        data prior to the validation chunk. (Which always belongs to the training
        data in the forecast cross validation scheme.) Otherwise, it takes the
        validation prior states from the beginning of the validation chunk.

        Args:
            val_chunk_start: The index of the first observation in the
                validation set.
            val_chunk_end: One more than the index of the last observation in
                the validation set.
            num_val_prior_states: The number of prior states to use.
            val_scheme: The validation scheme to use. One of ["forecast",
                "last", "all].

        Returns:
            val_prior_state_start: The index of the first observation in the
                validation prior states.
            val_prior_state_end: One more than the index of the last observation
                in the validation prior states.
        """
        if val_scheme == "forecast":
            val_prior_state_start = val_chunk_start - num_val_prior_states
            val_prior_state_end = val_chunk_start

        else:
            val_prior_state_start = val_chunk_start
            val_prior_state_end = val_chunk_start + num_val_prior_states

        # Add one so that the last prior state is the same as the first
        # prediction state as required by interfere.
        return val_prior_state_start + 1, val_prior_state_end + 1


    def _make_val_idxs(
        self,
        val_chunk_start: int,
        val_chunk_end: int,
        num_val_prior_states: int,
        val_scheme: str,
    ) -> Tuple[int, int]:
        """Build the indexes for the validation states.

        When the val_scheme is forecast, it takes validation states from the
        entire validation chunk. Otherwise, it takes the validation states from
        the portion of the validation chunk that excludes the prior states.

        Args:
            val_chunk_start: The index of the first observation in the
                validation chunk.
            val_chunk_end: One more than the index of the last observation in
                the validation chunk.
            num_val_prior_states: The number of prior states to use.
            val_scheme: The validation scheme to use. One of ["forecast",
                "last", "all].

        Returns:
            val_start_idx: The index of the first observation in the
                validation set.
            val_end_idx: One more than the index of the last observation in
                the validation set.
        """
        if val_scheme == "forecast":
            val_start_idx = val_chunk_start
            val_end_idx = val_chunk_end

        else:
            val_start_idx = val_chunk_start + num_val_prior_states
            val_end_idx = val_chunk_end

        return val_start_idx, val_end_idx
