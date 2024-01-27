import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
from typing import Tuple


# Разработчик модуля Кобылкин Константин Сергеевич

class ConstantModel:
    def __init__(self,
                 series: pd.DataFrame,
                 mean_window_size: int = 7,
                 mean_type: str = 'mean',
                 horizon: int = 12,
                 max_n_series_to_process_at_once: int = 40000,
                 backtest_horizon: int = 6,
                 backtest_depth: int = 12):
        self.mean_window_size = mean_window_size
        self.mean_type = mean_type
        self.series = series.copy().astype(np.float64)
        self.series.index = pd.to_datetime(self.series.index,
                                           format='%Y-%m-%d')
        self.series.index.name = (self.series.index.name
                                  if self.series.index.name else 'Date')
        self.series.columns.name = (self.series.columns.name
                                    if self.series.columns.name else 'series_names')
        self.horizon = horizon
        self.means = None
        self.max_n_series_to_process_at_once = max_n_series_to_process_at_once
        self.backtest_horizon = backtest_horizon
        self.backtest_depth = backtest_depth
        self.initial_series = self.series.copy()

    def fit(self):
        if self.mean_type == 'mean':
            self.means = self.series.iloc[- self.mean_window_size:, :].mean(axis=0)
        elif self.mean_type == 'median':
            self.means = self.series.iloc[- self.mean_window_size:, :].median(axis=0)

    def predictForPeriod(self,
                         forecasting_period: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(np.tile(self.means.values.reshape(1, -1),
                                    reps=(len(forecasting_period), 1)),
                            index=forecasting_period,
                            columns=self.series.columns)

    def predict(self):
        forecasting_period = pd.Series(pd.date_range(start=self.series.index[-1] + MonthEnd(1),
                                                     periods=self.horizon,
                                                     freq='M'))
        return self.predictForPeriod(forecasting_period=forecasting_period)

    def backtestPredict(self,
                        group: Tuple[pd.core.indexes.datetimes.DatetimeIndex,
                                     pd.DataFrame]) -> pd.DataFrame:
        latest_fact_date, series = group[0][-1], group[1]
        n_series_groups = len(group[0])
        self.series = pd.concat(tuple(pd.concat({str(i): series[:latest_fact_date].copy()
                                                                                  .shift(i)
                                                                                  .T},
                                                names=['shift']).T
                                      for i in range(n_series_groups)),
                                axis=1)
        self.fit()
        predictions = self.predict()

        horizon = predictions.shape[0]
        prediction_dates = pd.concat(tuple(pd.Series(predictions.index + MonthEnd(- i))
                                           for i in range(n_series_groups)),
                                     axis=0)
        last_fact_dates = pd.concat(tuple(pd.Series([last_fact_date] * horizon)
                                          for last_fact_date in group[0][::-1]),
                                    axis=0)
        predictions = pd.DataFrame(np.concatenate(tuple(np.split(predictions.values,
                                                                 indices_or_sections=n_series_groups,
                                                                 axis=1)),
                                                  axis=0),
                                   index=prediction_dates,
                                   columns=series.columns)

        predictions['last_fact_date'] = last_fact_dates.values
        predictions.index.name = date_index_name = series.index.name
        predictions = predictions.reset_index()
        predictions['lag'] = (predictions[date_index_name].dt.to_period('M') -
                              predictions['last_fact_date'].dt.to_period('M')).apply(lambda x: x.n)
        predictions = predictions.loc[predictions[date_index_name] <= series.index[-1]]
        return predictions

    def formSeriesBacktests(self):
        series_forecast_dates = self.series.index[- self.backtest_depth:]
        series_copy = self.series.copy()
        forecasting_horizon = self.horizon

        self.horizon = self.backtest_horizon

        n_dates_to_process_at_once = (min(self.max_n_series_to_process_at_once // series_copy.shape[1],
                                          len(series_forecast_dates))
                                      if series_copy.shape[1] <= self.max_n_series_to_process_at_once
                                      else 1)

        groups = list((pd.date_range(start=date + MonthEnd(- 1),
                                     end=date + MonthEnd(n_dates_to_process_at_once - 2),
                                     freq='M'),
                       series_copy)
                      for date in series_forecast_dates[::n_dates_to_process_at_once])

        predictions_list = []
        for group in groups:
            predictions_list.append(self.backtestPredict(group))

        self.series = series_copy
        self.horizon = forecasting_horizon

        return pd.concat(predictions_list,
                         axis=0)


class SlidingConstantModel(ConstantModel):
    def predictSinglePeriod(self) -> pd.DataFrame:
        forecasting_period = pd.Series(pd.date_range(start=self.series.index[-1] + MonthEnd(1),
                                                     periods=1,
                                                     freq='M'))
        return self.predictForPeriod(forecasting_period=forecasting_period)

    def fit(self):
        super().fit()

        for _ in range(self.horizon):
            forecasting_date = self.series.index[-1] + MonthEnd(1)
            self.series.loc[forecasting_date, :] = self.predictSinglePeriod().values.reshape(-1)
            super().fit()

    def predict(self):
        predictions = self.series.iloc[-self.horizon:].copy()
        self.series = self.initial_series.copy()
        return predictions


