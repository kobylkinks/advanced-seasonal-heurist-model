import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
from tqdm import tqdm
from typing import Dict, List, Tuple


class BestModelChooser:
    def __init__(self,
                 models_zoo: Dict,
                 group_columns_to_estimate_mape: List,
                 series: pd.DataFrame,
                 horizon: int = 12,
                 backtest_depth: int = 12,
                 backtest_horizon: int = 6,
                 max_n_series_to_process_at_once: int = 40000,
                 percentage_to_select_best_models: float = 15,
                 additive_percentage_to_select_best_models: float = 2.0):
        self.best_models = None
        self.divisions_for_model = None
        self.original_columns = None
        self.models_errors = None
        self.backtest_columns = None
        self.fact_columns = None
        self.models_backtests = None
        self.backtest_depth = backtest_depth
        self.backtest_horizon = backtest_horizon
        self.horizon = horizon
        self.max_n_series_to_process_at_once = max_n_series_to_process_at_once
        self.percentage_to_select_best_models = percentage_to_select_best_models
        self.additive_percentage_to_select_best_models = additive_percentage_to_select_best_models

        models_zoo_ = models_zoo.copy()
        for model_name in models_zoo_:
            models_zoo_[model_name]['model_config']['backtest_depth'] = backtest_depth
            models_zoo_[model_name]['model_config']['backtest_horizon'] = backtest_horizon
            models_zoo_[model_name]['model_config']['horizon'] = horizon
            models_zoo_[model_name]['model_config']['max_n_series_to_process_at_once'] = \
                max_n_series_to_process_at_once

        self.models = {model_name: {'model_type': models_zoo_[model_name]['model_type'],
                                    'model_config': models_zoo_[model_name]['model_config']}
                       for model_name in models_zoo_}
        self.series = series.copy()
        self.series.index = pd.to_datetime(self.series.index,
                                           format='%Y-%m-%d')
        self.series.index.name = (self.series.index.name
                                  if self.series.index.name else 'Date')
        self.series.columns.name = (self.series.columns.name
                                    if self.series.columns.name else 'series_names')
        self.group_columns_to_estimate_mape = group_columns_to_estimate_mape

    def formModelBacktests(self,
                           model_name: str) -> pd.DataFrame:
        date_column_name = self.series.index.name
        model_config = self.models[model_name]['model_config']
        model = self.models[model_name]['model_type'](series=self.series,
                                                      **model_config)
        model_backtests = model.formSeriesBacktests() \
            .drop(['last_fact_date'],
                  axis=1) \
            .set_index(['lag',
                        date_column_name])
        model_backtests = model_backtests.T \
            .groupby(self.group_columns_to_estimate_mape) \
            .agg(func=np.nansum) \
            .T
        backtests = pd.concat({'forecast': model_backtests.T},
                              names=['data_type']).T
        self.backtest_columns = backtests.columns
        return backtests.reset_index()

    def formModelsBacktests(self):
        models_names = list(self.models.keys())
        self.models_backtests = {}
        print('Generating models backtests')
        for model_name in tqdm(models_names):
            self.models_backtests[model_name] = self.formModelBacktests(model_name)
        self.models_backtests = pd.concat(self.models_backtests,
                                          names=['model_name']).reset_index() \
            .drop(['level_1'],
                  axis=1)

    def computeLagErrors(self,
                         group: Tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        models_forecasts, facts = group[0], group[1]
        date_column_name = self.series.index.name
        dates = (set(models_forecasts[date_column_name].unique()) &
                 set(facts[date_column_name].unique()))

        facts_ = facts.loc[facts[date_column_name].isin(dates)].sort_values([date_column_name])

        models_forecasts_ = models_forecasts.loc[models_forecasts[date_column_name].isin(dates)] \
            .sort_values(['model_name', date_column_name]) \
            .reset_index(drop=True)
        n_models = len(self.models)

        combined = pd.concat((models_forecasts_,
                              pd.concat([facts_] * n_models,
                                        axis=0).reset_index(drop=True)),
                             axis=1)
        errors = (np.abs(combined[self.backtest_columns].values -
                         combined[self.fact_columns].values) /
                  np.abs(combined[self.fact_columns]).values) * 100
        errors = pd.DataFrame(errors,
                              columns=list(self.original_columns))
        errors = pd.concat((errors,
                            combined[['model_name', date_column_name]]),
                           axis=1)
        return errors.groupby(('model_name',) +
                              ('',) * self.original_columns.nlevels)[self.original_columns] \
            .agg(lambda x: x[- self.backtest_horizon:].mean(axis=0))

    def computeAverageErrors(self):
        aggregated_series = self.series.T \
            .groupby(self.group_columns_to_estimate_mape) \
            .agg(func=np.nansum) \
            .T
        self.original_columns = aggregated_series.columns
        aggregated_series = pd.concat({'fact': aggregated_series.T},
                                      names=['data_type']).T
        self.fact_columns = aggregated_series.columns
        aggregated_series = aggregated_series.reset_index()

        print('Computing average errors')
        models_errors_list = []
        for i in tqdm(range(1, self.backtest_horizon + 1)):
            forecast_fact_data = (self.models_backtests.loc[self.models_backtests.lag == i],
                                  aggregated_series)
            models_errors_list.append(self.computeLagErrors(forecast_fact_data))
        self.models_errors = pd.concat(models_errors_list,
                                       axis=0).groupby(('model_name',) +
                                                       ('',) * self.original_columns.nlevels)[self.original_columns] \
            .mean() \
            .T

    def chooseBestModels(self):
        factor_to_choose_models_for_ensemble = 1 + self.percentage_to_select_best_models / 100
        best_models = (self.models_errors.values <= (factor_to_choose_models_for_ensemble *
                                                     self.models_errors.min(axis=1).values.reshape(-1, 1)))
        best_models = (best_models |
                       (self.models_errors.values <= self.models_errors.min(axis=1).values.reshape(-1, 1) +
                        self.additive_percentage_to_select_best_models))

        self.best_models = pd.DataFrame(best_models,
                                        columns=self.models_errors.columns,
                                        index=self.models_errors.index)
        self.divisions_for_model = {}

        print('Choosing best models')
        for model_name in tqdm(self.models):
            self.divisions_for_model[model_name] = set(self.best_models.index[self.best_models[model_name]])
        self.divisions_for_model['nan'] = set(self.best_models.index[self.best_models.sum(axis=1) == 0])

    def fit(self):
        self.formModelsBacktests()
        self.computeAverageErrors()
        self.chooseBestModels()

    def predict(self) -> pd.DataFrame:
        all_predictions_list = []

        reordered_level_names = (self.group_columns_to_estimate_mape +
                                 list(set(self.series.columns.names) -
                                      set(self.group_columns_to_estimate_mape)))

        for model_name in tqdm(self.models):
            cols = list(col for col in self.series.columns.reorder_levels(reordered_level_names)
                        if col[:len(self.group_columns_to_estimate_mape)] in self.divisions_for_model[model_name])
            cols = pd.MultiIndex.from_tuples(cols,
                                             names=reordered_level_names) \
                .reorder_levels(self.series.columns.names)

            model_config = self.models[model_name]['model_config']
            model = self.models[model_name]['model_type'](series=self.series[cols],
                                                          **model_config)
            model.fit()
            predictions = model.predict()
            all_predictions_list.append(predictions)

        all_predictions = pd.concat(all_predictions_list,
                                    axis=1).T \
            .groupby(self.series.columns.names) \
            .mean() \
            .T
        return all_predictions[self.series.columns]