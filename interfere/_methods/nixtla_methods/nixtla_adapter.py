from abc import abstractmethod
import logging
from math import ceil
import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import neuralforecast
from neuralforecast.common._base_model import BaseModel as NeuralForecastBaseModel
import numpy as np
import pandas as pd
import statsforecast
from statsforecast.models import _TS as StatsForecastBaseModel

from ...base import ForecastMethod
from ...base import DEFAULT_RANGE
from ...utils import copy_doc


class NixtlaAdapter(ForecastMethod):
    """Adapter that bridges nixtla and interfere predictive methods.

    Notes: Inheriting classes must define an __init__ function that passes these
    three arguments to the __init__ function below:
        (1) `method_type` which must be subtype of
            `neuralforecast.models.common.BaseModel` or be a class from `statsforecast.models`.
        (2) `method_params` which is an instance of a `Dict[str, Any]`.
        (3) `nixtla_forecaster_class` one of 
            [statsforecast.StatsForecast, neuralforecast.NeuralForecast].

    For example

    ```
    class LSTM(NixtlaAdapter)
        def __init__(self, a, b, c=1):
            method_params = {"a": a, "b": b, "c": c}
            method_type = neuralforecast.models.LSTM
            nixtla_forecaster_class = neuralforecast.NeuralForecast
            super().__init__(
                method_type,
                method_params,
                nixtla_forecaster_class
            )
            
    ```

    Inhereting classes must also provide definitions for all of the following
    methods: 
        
    def get_window_size(self):
        '''Returns how many historic time steps are needed to make a prediction.'''
    

    def get_horizon(self):
        '''Returns how many timesteps the method will predict.'''


    def get_test_params() -> Dict[str, Any]:
        '''Returns default parameters conducive to fast test cases'''    
    """

    def __init__(
        self,
        method_type: Union[
            Type[NeuralForecastBaseModel],
            Type[StatsForecastBaseModel],
        ],
        method_params: Dict[str, Any],
        nixtla_forecaster_class: Union[
            Type[neuralforecast.NeuralForecast], 
            Type[statsforecast.StatsForecast]
        ]
    ):
        """Initializes a Nixtla <-> Interfere adapter.

        Args:
            method_type (Nixtla Base Model): The type of Nixtla model to be 
                adapted to interfere.
            method_params (Dict[str, Any]): The Nixtla model's parameters.
            nixtla_forecaster_class (Type): one of 
            [statsforecast.StatsForecast, neuralforecast.NeuralForecast].
        """
        self.method_type = method_type
        self.method_params = method_params
        self.nixtla_forecaster_class = nixtla_forecaster_class

        # The following args prevent pytorch lightning from 
        # writing to files in order to enable parallel grid search.
        self.set_params(
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False
        )

        # Turn off logs.
        logging.getLogger(
            "pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
        logging.getLogger(
            "pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
        logging.getLogger(
            "lightning_fabric.utilities.seed").setLevel(logging.WARNING)
        logging.getLogger(
            "pytorch_lightning.callbacks.model_summary").setLevel(
                logging.WARNING)
        logging.getLogger(
            "pytorch_lightning.trainer.connectors.signal_connector").setLevel(
                logging.WARNING)


    @copy_doc(ForecastMethod._fit)
    def _fit(
        self,
        t: np.ndarray,
        endog_states: np.ndarray,
        exog_states: np.ndarray = None
    ):
        self.dt_ = t[1] - t[0]

        if not np.allclose(np.diff(t), self.dt_, atol=1e9):
            raise ValueError(
                "Nixtla forecasters require evenly spaced time points.")

        if exog_states is not None:
            _, k = exog_states.shape
        else:
            k = 0
    
        # Assign names to exogeneous variables.
        self.exog_state_ids = default_exog_names(k)

        # Convert time to discrete.
        discrete_t = self.to_discrete(t)

        # Create a neuralforecast compatible DataFrame.
        train_df = to_nixtla_df(
            discrete_t,
            endog_states,
            exog_states,
            exog_state_ids=self.exog_state_ids
        )
        
        # Initialize model.
        self.model = self.method_type(**self.method_params)
        # Initialize neural forecaster.
        self.nixtla_forecaster = self.nixtla_forecaster_class(
            models=[self.model],
            freq="s"
        )
        self.nixtla_forecaster.fit(df=train_df)


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
        
        if len(prior_t) < self.get_window_size():
            raise ValueError("Not enough context provided for to make a "
                f"prediction. {len(prior_t)} obs provided, need "
                f"{self.get_window_size()}."
            )
        
        pred_dt = t[1] - t[0]
        if not np.allclose(self.dt_, pred_dt, atol=1e9):
            raise ValueError(
                f"{str(type(self).__name__)}.predict() times must have the same"
                " step size as the time values passed to .fit()"
            )
        
        if isinstance(self.nixtla_forecaster, neuralforecast.NeuralForecast):
            return self.neuralforecast_predict(
                t,
                prior_endog_states,
                prior_exog_states,
                prior_t,
                prediction_exog,
                rng,
            )
        
        if isinstance(self.nixtla_forecaster, statsforecast.StatsForecast):
            return self.statsforecast_predict(
                t,
                prior_endog_states,
                prior_exog_states,
                prior_t,
                prediction_exog,
                rng,
            )

        
    @copy_doc(ForecastMethod._predict)
    def neuralforecast_predict(
        self,
        t: np.ndarray,
        prior_endog_states: np.ndarray,
        prior_exog_states: Optional[np.ndarray] = None,
        prior_t: Optional[np.ndarray] = None,
        prediction_exog: Optional[np.ndarray] = None,
        rng: np.random.RandomState = DEFAULT_RANGE,
    ) -> np.ndarray:
        
        # Set environment variable to adopt future behavior where predict
        # returns ID as a column. It will be safe to remove this after Nixtla
        # updates neuralforecast and adopts this behavior as the default.
        os.environ['NIXTLA_ID_AS_COL'] = "1"
        
        # Convert to discrete time.
        discrete_t = self.to_discrete(t)

        # Total number of predictions times. First time corresponds to current
        # state. 
        m_pred = len(discrete_t) - 1

        # Internal forecasting method's prediction horizon.
        h = self.get_horizon()

        # How many recursive predictions the model will need to do.
        n_steps = ceil(m_pred / h)

        # Number of additional exog datapoints needed to fill the
        # prediction horizon for the last prediction chunk.
        # Add one because no prediction is needed for the first timestep.
        n_extra_preds = n_steps * h - m_pred + 1

        # Names of exogeneous variables.
        exog_state_ids = self.exog_state_ids

        # Initial historical context DataFrame.
        df = to_nixtla_df(
            self.to_discrete(prior_t),
            prior_endog_states,
            prior_exog_states,
            exog_state_ids=exog_state_ids
        )
        
        # Build times and signals for the exogeneous input data frame.
        futr_times = np.hstack([
            discrete_t,
            np.arange(1, n_extra_preds + 1) + discrete_t[-1]
        ])

        if prediction_exog is None:
            prediction_exog = np.full(
                (len(discrete_t), len(exog_state_ids)),
                np.inf
            )
        
        # Repeat last row of exogeneous for any extra predictions required
        # due to forcast horizon window size.
        futr_exog = np.vstack([prediction_exog] + [
            prediction_exog[-1] for _ in range(n_extra_preds)
        ])  

        futr_df = to_nixtla_df(
            futr_times,
            exog_states=futr_exog,
            unique_ids=[id for id in df.unique_id.unique()],
            exog_state_ids=exog_state_ids
        )

        # Run recursive predictions.
        for _ in range(n_steps):
            # Make prediction.
            pred_df = self.nixtla_forecaster.predict(df=df, futr_df=futr_df)
            
            # Reformat prediction.
            pred_df = pred_df.rename(columns={str(self.model): "y"})
            # Add in exogeneous states
            merged_df = pd.merge(
                pred_df, 
                futr_df[["ds"] + exog_state_ids].drop_duplicates(), 
                on="ds"
            )
            # Append to historical context.
            df = pd.concat([df, merged_df])

        # Convert to endogenous array.
        _, endog_pred, _ = to_interfere_arrays(
            df,
            unique_ids=[id for id in df.unique_id.unique()]
        )

        # Remove historic observations
        endog_pred = endog_pred[(len(prior_t) - 1):, :]
        # Remove extra predictions.
        endog_pred = endog_pred[:(m_pred + 1), :]
        return endog_pred


    @copy_doc(ForecastMethod._predict)
    def statsforecast_predict(
        self,
        t: np.ndarray,
        prior_endog_states: np.ndarray,
        prior_exog_states: Optional[np.ndarray] = None,
        prior_t: Optional[np.ndarray] = None,
        prediction_exog: Optional[np.ndarray] = None,
        rng: np.random.RandomState = DEFAULT_RANGE,
    ) -> np.ndarray:
        # Set environment variable to adopt future behavior where predict
        # returns ID as a column. It will be safe to remove this after Nixtla
        # updates neuralforecast and adopts this behavior as the default.
        os.environ['NIXTLA_ID_AS_COL'] = "1"
        
        # Convert to discrete time.
        discrete_t = self.to_discrete(t)

        # Total number of predictions times. First time corresponds to current
        # state. 
        m_pred = len(discrete_t) - 1

        # Internal forecasting method's prediction horizon.
        h = self.get_horizon()

        # How many recursive predictions the model will need to do.
        n_steps = ceil(m_pred / h)

        # Number of additional exog datapoints needed to fill the
        # prediction horizon for the last prediction chunk.
        # Add one because no prediction is needed for the first timestep.
        n_extra_preds = n_steps * h - m_pred + 1

        # Names of exogeneous variables.
        exog_state_ids = self.exog_state_ids

        # Initial historical context DataFrame.
        df = to_nixtla_df(
            self.to_discrete(prior_t),
            prior_endog_states,
            prior_exog_states,
            exog_state_ids=exog_state_ids
        )
        
        # Build times and signals for the exogeneous input data frame.
        futr_times = np.hstack([
            discrete_t,
            np.arange(1, n_extra_preds + 1) + discrete_t[-1]
        ])

        if prediction_exog is None:
            prediction_exog = np.full(
                (len(discrete_t), len(exog_state_ids)),
                np.inf
            )
        
        # Repeat last row of exogeneous for any extra predictions required
        # due to forcast horizon window size.
        futr_exog = np.vstack([prediction_exog] + [
            prediction_exog[-1] for _ in range(n_extra_preds)
        ])  

        X_df = to_nixtla_df(
            futr_times,
            exog_states=futr_exog,
            unique_ids=[id for id in df.unique_id.unique()],
            exog_state_ids=exog_state_ids
        )

        # Run recursive predictions.
        for i in range(1, n_steps + 1):

            X_df = to_nixtla_df(
            futr_times[i],
            exog_states=futr_exog[i:i+1, :],
            unique_ids=[id for id in df.unique_id.unique()],
            exog_state_ids=exog_state_ids
        )
            # Make prediction.
            pred_df = self.nixtla_forecaster.forecast(h=1, df=df, X_df=X_df)
            
            # Reformat prediction.
            pred_df = pred_df.rename(columns={str(self.model): "y"})
            # Add in exogeneous states
            merged_df = pd.merge(
                pred_df, 
                X_df[["ds"] + exog_state_ids].drop_duplicates(), 
                on="ds"
            )
            # Append to historical context.
            df = pd.concat([df, merged_df])

        # Convert to endogenous array.
        _, endog_pred, _ = to_interfere_arrays(
            df,
            unique_ids=[id for id in df.unique_id.unique()]
        )

        # Remove historic observations
        endog_pred = endog_pred[(len(prior_t) - 1):, :]
        # Remove extra predictions.
        endog_pred = endog_pred[:(m_pred + 1), :]
        return endog_pred
        
        h = len(t) - 1
        lag = self.get_window_size()

        # Grab only the neccesary number of historic obs.
        prior_t = prior_t[-lag:]
        prior_endog_states = prior_endog_states[-lag:, :]

        if prior_exog_states is not None:
            prior_exog_states = prior_exog_states[-lag:, :]

        #  Historical context DataFrame.
        df = to_nixtla_df(
            prior_t,
            prior_endog_states,
            prior_exog_states,
            exog_state_ids=self.exog_state_ids
        )

        if prediction_exog is not None:
            # Translate exogeneous data to a dataframe.
            X_df = to_nixtla_df(
                t,
                exog_states=prediction_exog,
                exog_state_ids=self.exog_state_ids,
                unique_ids=[id for id in df.unique_id.unique()]
            )
        else:
            X_df = None

        # Make a prediction.
        pred_df = self.nixtla_forecaster.forecast(df=df, h=h, X_df=X_df)

        # Reshape prediction dataframe.
        pred_df = pred_df.reset_index()
        pred_df = pred_df.rename(columns={str(self.model): "y"})

        # Convert to interfere arrays.
        _, endog_pred, _ = to_interfere_arrays(
            pred_df,
            unique_ids=[id for id in pred_df.unique_id.unique()]
        )

        # Include the initial state of the system.
        endog_pred = np.vstack([prior_endog_states[-1, :], endog_pred])

        return endog_pred
    
    
    def to_discrete(self, t: np.ndarray):
        """Converts an evenly spaced time array to discrete time."""
        dt = t[1] - t[0]

        if not np.allclose(np.diff(t), dt, atol=1e9):
            raise ValueError("Nixtla Forecasters require evenly spaced times.")
        
        if not np.allclose(self.dt_, dt, atol=1e9):
            raise ValueError(
                "The step size of the time array passed to "
                f"{type(self).__name__}.fit() must be the same as the time step"
                f" of the time arrays passed to {(self).__name__}.predict()."
            )
        
        start_int = np.round(t[0] / dt, decimals=0)
        end_int = start_int + len(t)
        discrete_t = np.arange(start_int, end_int)
        return discrete_t


    def set_params(self, **params):
        self.method_params = {**self.method_params, **params}
        self.model = self.method_type(**self.method_params)


    def get_params(self, deep: bool = True) -> Dict:
        return self.method_params
    
    
    @abstractmethod
    def get_horizon(self):
        """Returns the number of timesteps forecasted by the method.
        """
        raise NotImplementedError()
    

def default_exog_names(num_exog_vars: int):
    """Makes default names for exogeneous variables."""
    return [f"u{i}" for i in range(num_exog_vars)]


def to_nixtla_df(
    t: np.ndarray,
    endog_states: np.ndarray = None,
    exog_states: np.ndarray = None,
    unique_ids: List[Any] = None,
    exog_state_ids: List[Any] = None
):
    """Transforms interfere time series arrays to a neuralforcast dataframe.
    
    Args:
        t: An (m,) array of time points.
        endog_states: An (m, n) array of endogenous signals. Sometimes
            called Y. Rows are observations and columns are variables. Each
            row corresponds to the times in `t`.
        exog_states: An (m, k) array of exogenous signals. Sometimes called
            X. Rows are observations and columns are variables. Each row 
            corresponds to the times in `t`.
        unique_ids: A list of identifiers for each endogenous signal.
            E.g. `["x0", "x1", "x2"]`.
        exog_state_ids: A list of identifiers for each exogenous signal.
            E.g. `["u0", "u1"]`.
    """
    col_names = []

    if (endog_states is None) and (unique_ids is None):
        raise ValueError("unique ids cannot be determined unless one of "
                            "`endog_states` and `unique_ids` is not None.")
    
    if unique_ids is None:
        _, n = endog_states.shape
        # Default unique_ids:
        unique_ids = [f"x{i}" for i in range(n)]

    # Associate forecast times with each unique ID.
    data_chunks = [
        np.vstack([
            t, np.full(t.shape, id),
        ]).T
        for id in unique_ids
    ]
    col_names += ["ds", "unique_id"]

    # Optionally add endogeneous states
    if endog_states is not None:
        data_chunks = [
            np.hstack([X, y.reshape(-1, 1)])
            for X, y in zip(data_chunks, endog_states.T)
        ]
        col_names += ["y"]

    if exog_states is None:
        if not ((exog_state_ids is None) or (exog_state_ids == [])):
            raise ValueError("Exogengous state ids passed without "
                "accompanying exogeneous states. Supply a value to "
                "`exog_states` argument.")
        if exog_state_ids is None:
            exog_state_ids = []
    else:
        # Add exogeneous states to data
        data_chunks = [
            np.hstack([x, exog_states]) for x in data_chunks
        ]

        if exog_state_ids is None:
            # Assign default exogeneous state ids when not provided.
            _, k = exog_states.shape
            exog_state_ids = default_exog_names(k)

        col_names += exog_state_ids

    # Stack arrays into a dataframe.
    nf_data = pd.DataFrame(
        np.vstack(data_chunks), columns=col_names
    )

    # Translate float time to datetime.
    nf_data.ds = pd.to_datetime(
        pd.to_numeric(nf_data.ds), unit='s', errors='coerce')

    # Transform numeric columns.
    for c in nf_data.columns:
        if c in ["y"] + exog_state_ids:
            nf_data[c] = pd.to_numeric(nf_data[c], errors='coerce')

    return nf_data
    

def to_interfere_arrays(
    nixtla_df: pd.DataFrame,
    unique_ids: Optional[List[Any]] = None,
    exog_state_ids: Optional[List[Any]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a nixtla dataframe to interfere arrays.
    
    Args:
        nueralforecast_df: A data frame with columns
        unique_ids: A list of ids corresponding to endogenous variables, in
            column order.
        exog_state_ids: A list of exogenous variable ids in column order. 

    Returns:
        t: An (m,) array of time points.
        endog_states: An (m, n) array of endogenous signals. Sometimes
            called Y. Rows are observations and columns are variables. Each
            row corresponds to the times in `t`.
        exog_states: An (m, k) array of exogenous signals. Sometimes called
            X. Rows are observations and columns are variables. Each row 
            corresponds to the times in `t`.
    """
    if unique_ids is None:
        
        unique_ids = nixtla_df.unique_id.unique()
        dflt_unique_ids = [f"x{i}" for i in range(len(unique_ids))]

        if len(set(unique_ids).symmetric_difference(dflt_unique_ids)) > 0:
            raise ValueError("No value passed to `unique_id` argument but " 
                                "default unique IDs were not used. Supply "
                                "endogenous column names in order to "
                                "`unique_id` argument." )
        
        unique_ids = dflt_unique_ids

    
    if exog_state_ids is None:
        exog_state_ids = [
            col_name for col_name in nixtla_df.columns
            if col_name not in ["unique_id", "ds", "y"]
        ]

    # Extract endog variables.
    endog_states = np.vstack([
        nixtla_df[nixtla_df.unique_id == id].y
        for id in unique_ids
    ]).T

    # Grab exogeneous.
    x0_mask = nixtla_df.unique_id == unique_ids[0]
    exog_states = nixtla_df[x0_mask][exog_state_ids]

    # Convert ds to float.
    t = nixtla_df[x0_mask].ds.values.astype(float) / 1_000_000_000

    return t, endog_states, exog_states