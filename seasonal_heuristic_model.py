import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
from typing import Dict, Tuple
from workalendar.europe import Russia




class BasicSeasonalHeuristicModel:
    def __init__(self, series: pd.DataFrame,
                 horizon: int = 12,
                 adjust_for_day_count: bool = False,
                 days_type: str = 'all',
                 smooth_series: bool = False,
                 smoothing_window_size: int = 9):
        self.horizon = horizon
        self.adjust_for_day_count = adjust_for_day_count
        self.days_type = days_type

        self.original_series = series.copy().astype(np.float64)
        self.original_series.index = pd.to_datetime(self.original_series.index,
                                                    format='%Y-%m-%d')
        self.original_series.index.name = (self.original_series.index.name
                                           if self.original_series.index.name else 'Date')
        self.original_series.columns.name = (self.original_series.columns.name
                                             if self.original_series.columns.name else 'series_names')

        adjusted_series = self.adjustSeriesForDaysCounts(self.original_series)
        if smooth_series:
            self.series = adjusted_series.rolling(window=smoothing_window_size,
                                                  min_periods=1,
                                                  center=False,
                                                  axis=0).median()
        else:
            self.series = adjusted_series

    def computeDaysCounts(self, dates: pd.DatetimeIndex) -> pd.Series:
        russian_calendar = Russia()
        ts_dates = pd.Series(dates)
        ts_dates = pd.to_datetime(ts_dates,
                                  format='%Y-%m-%d')
        days_counts = ts_dates.apply(lambda x_date: x_date.day)
        working_days_counts = ts_dates.apply(lambda x_date:
                                             russian_calendar.get_working_days_delta(start=x_date + MonthEnd(-1),
                                                                                     end=x_date))
        holidays_counts = days_counts - working_days_counts

        if self.days_type == 'all':
            return days_counts
        elif self.days_type == 'working':
            working_days_counts[working_days_counts == 0] = np.nan
            return working_days_counts
        elif self.days_type == 'holidays':
            holidays_counts[holidays_counts == 0] = np.nan
            return holidays_counts
        elif self.days_type is None:
            return pd.Series(np.ones(shape=(len(ts_dates), )))

    def adjustSeriesForDaysCounts(self, series: pd.DataFrame) -> pd.DataFrame:
        if self.adjust_for_day_count:
            days_counts = self.computeDaysCounts(series.index)
            return pd.DataFrame(series.values /
                                days_counts.values.reshape(-1, 1),
                                columns=series.columns,
                                index=series.index)
        else:
            return series

    def fit(self):
        pass

    def predict(self) -> pd.DataFrame:
        pass


class SimpleSeasonalHeuristicModel(BasicSeasonalHeuristicModel):
    
    """Класс реализует сезонную эвристическую модель прогноза временного ряда. 
       Расчет прогноза можно пояснить на следующем частном случае. Предположим, что прогноз
       делается на август 2023 года при условии, что последний фактический месяц - 
       июнь 2023.       
       
       прогноз на август = (средний прирост значений ряда за год * 
                            фактическое значение ряда в августе 2022 +
                            средний прирост значений ряда за 2 года *
                            фактическое значение ряда в августе 2021) / 2,
       где средний прирост значений ряда за год = (факт апреля 2023 / факт апреля 2022 +
                                                   факт мая 2023 / факт мая 2022 +
                                                   факт июня 2023 / факт июня 2022) / 3,
           средний прирост значений ряда за 2 года = (факт апреля 2023 / факт апреля 2021 +
                                                      факт мая 2023 / факт мая 2021 +
                                                      факт июня 2023 / факт июня 2021) / 3.
       Здесь средние годовые приросты - одни и те же для всех прогнозных месяцев.

       Разработчик модуля Кобылкин Константин Сергеевич, ЦК ПИМОП, Сбербанк

       """
              
    def __init__(self, n_years: int = 2, n_months: int = 3,
                 almost_zero_value: float = 0.1,
                 averaging_method: Dict = {'growth_factors': 'mean', 
                                           'forecast': 'mean'},
                 use_growth_factor_censoring: bool = False,
                 growth_factor_lower_threshold_quantile: float = 0.05,
                 growth_factor_upper_threshold_quantile: float = 0.95,
                 max_n_series_to_process_at_once: int = 40000,
                 backtest_horizon: int = 6,
                 backtest_depth: int = 12,
                 *args, **kwargs):
        
        """Конструктор класса, в котором задаются все гиперпараметры для расчета.
    
           ----------------------------------------------------------------------------------------------------------
           
           Параметры:

           series: pd.DataFrame - датафрейм с временными рядами со следующей структурой:
           каждому временному ряду должен соответствовать отдельный столбец в датафрейме,
           при этом в качестве индекса датафрейма должны использоваться (отсортированные
           по возрастанию) даты в формате pandas Timestamp
           (см. функцию pd.to_datetime(*, format='%Y-%m-%d')).

           Пример подготовки series:
           series = data.pivot(columns=['TB','GOSB'],
                               index=['dt'],
                               values=[target_name]) или
           series = data.pivot_table(columns=['TB','GOSB'],
                                     index=['dt'],
                                     values=[target_name],
                                     aggfunc='sum')

           horizon: int - горизонт прогнозирования (не должен превышать 12);

           adjust_for_day_count: bool - значения временных рядов делятся на число всех (рабочих или выходных)
           дней в месяце: например, значение августа 2023 каждого из временных рядов делится на число всех
           (рабочих или выходных) дней в августе 2023, значение июля 2023 делится на число дней в июле и т.д.
           Далее прогноз делается для вычисленных временных рядов из отношений согласно описанным выше формулам,
           полученные прогнозы умножаются на количество дней в прогнозных месяцах;

           days_type: str - используемое количество дней: 'all' - все дни, 'working' - рабочие дни и 'holidays' -
           выходные дни.
        
           n_years: int - число лет, используемое в расчете прогноза (в примере выше n_years = 2);

           n_months: int - число месяцев вплоть до последнего фактического месяца, используемых 
           в расчете прогноза (в примере выше n_months = 3);

           averaging_method: Dict - задает методы усреднения, используемые при расчете средних приростов
           ряда и самого прогноза (в примере выше averaging_method = {'growth_factors': 'mean', 'forecast':'mean'}). 
           Также возможно применить устойчивое к выбросам усреднение на основе среднего геометрического по формуле:
           среднее = корень k-й степени из произведения k отдельных значений, где либо k=n_years, либо k=n_months;

           use_growth_factor_censoring: bool - цензурировать или нет выбросы по месячным приростам.
           Например, если отношение факт апреля 2023 / факт апреля 2022 для какого то отдельного ряда
           превышает 99-й процентиль значений всех отношений апреля 2023 к апрелю 2022, вычисленных по всем временным
           рядам (прогноз предполагается делать сразу для многих временных рядов), то это отношение заменяется
           на значение 99-й процентиля в расчете среднего прироста значений ряда за год;
           
           growth_factor_lower_threshold_quantile: float - число между 0 и 1, задающее нижний процентиль значений
           для цензурирования выбросов (см. описание параметра use_growth_factor_censoring). Применимо
           только если use_growth_factor_censoring = True;

           growth_factor_upper_threshold_quantile: float - число между 0 и 1, задающее верхний процентиль значений
           для цензурирования выбросов (см. описание параметра use_growth_factor_censoring). Применимо
           только если use_growth_factor_censoring = True;

        """
        super().__init__(*args, **kwargs)
        self.initial_series = self.series.copy()
        self.n_years = n_years
        self.n_months = n_months
        self.averaging_method = averaging_method
        self.use_growth_factor_censoring = use_growth_factor_censoring
        self.growth_factor_lower_threshold_quantile = growth_factor_lower_threshold_quantile
        self.growth_factor_upper_threshold_quantile = growth_factor_upper_threshold_quantile
        self.almost_zero_value = almost_zero_value
        self.growth_factors = None
        self.n_series_groups = 1
        self.max_n_series_to_process_at_once = max_n_series_to_process_at_once
        self.backtest_horizon = backtest_horizon
        self.backtest_depth = backtest_depth

    def generateDatesForAllYears(self, dates: pd.Series,
                                 only_previous_years: bool = False):
        dates_for_all_years = pd.concat((dates + MonthEnd(i * 12)
                                         for i in range(- self.n_years, 
                                                        1 if not only_previous_years else 0)),
                                        axis=0)
        return np.array(dates_for_all_years)

    def censorFactors(self,
                      factors: np.ndarray,
                      factor_lower_threshold_quantile: float,
                      factor_upper_threshold_quantile: float) -> np.ndarray:
        splits = np.split(factors,
                          indices_or_sections=self.n_series_groups,
                          axis=-1)
        grouped_factors = np.concatenate(tuple(np.expand_dims(split,
                                                              axis=1)
                                               for split in splits),
                                         axis=1)
        factor_lower_thresholds = np.tile(
            np.nanquantile(grouped_factors,
                           q=factor_lower_threshold_quantile,
                           axis=-1,
                           keepdims=True),
            reps=(1, ) * factors.ndim + (self.series.shape[1] // self.n_series_groups, )
        )
        factor_lower_thresholds = np.concatenate(tuple(factor_lower_thresholds[:, i, ...]
                                                       for i in range(self.n_series_groups)),
                                                 axis=-1)

        factor_upper_thresholds = np.tile(
            np.nanquantile(grouped_factors,
                           q=factor_upper_threshold_quantile,
                           axis=-1,
                           keepdims=True),
            reps=(1, ) * factors.ndim + (self.series.shape[1] // self.n_series_groups, )
        )
        factor_upper_thresholds = np.concatenate(tuple(factor_upper_thresholds[:, i, ...]
                                                       for i in range(self.n_series_groups)),
                                                 axis=-1)

        factors = (((factors <= factor_upper_thresholds) &
                    (factors >= factor_lower_thresholds)) *
                   factors +
                   (factors > factor_upper_thresholds) *
                   factor_upper_thresholds +
                   (factors < factor_lower_thresholds) *
                   factor_lower_thresholds)
        return factors

    def getAllYearsValues(self) -> np.ndarray:
        latest_fact_dates = pd.Series(self.series.index[- self.n_months:])
        dates_for_all_years = self.generateDatesForAllYears(latest_fact_dates)
        return np.array(self.series.loc[dates_for_all_years])

    def getValuesRatios(self):
        all_years_vals = self.getAllYearsValues()
        ratio_denominators = all_years_vals[:- self.n_months, :]
        ratio_denominators[np.abs(ratio_denominators) <= self.almost_zero_value] = np.nan
        return (np.tile(all_years_vals[- self.n_months:, :],
                        reps=(self.n_years, 1)) /
                ratio_denominators)

    def fitGrowthFactors(self):
        growth_factors = self.getValuesRatios()

        if self.use_growth_factor_censoring:
            growth_factors = self.censorFactors(
                factors=growth_factors,
                factor_lower_threshold_quantile=self.growth_factor_lower_threshold_quantile,
                factor_upper_threshold_quantile=self.growth_factor_upper_threshold_quantile
            )

        if self.averaging_method['growth_factors'] == 'geometric mean':
            growth_factors = np.log(growth_factors + 1)
        
        growth_factors = np.array(pd.DataFrame(np.concatenate((np.tile(np.arange(self.n_years, 0, -1).reshape(-1, 1),
                                                                       reps=(1, self.n_months)).reshape(-1, 1),
                                                               growth_factors),
                                                              axis=1),
                                               columns=['year'] +
                                               list(self.series.columns)).groupby('year')
                                                                         .agg(func=np.nanmean)
                                                                         .sort_index(ascending=False))
        self.growth_factors = (np.exp(growth_factors) - 1
                               if self.averaging_method['growth_factors'] == 'geometric mean'
                               else growth_factors)

    def fit(self):
        
        """В зависимости от значения параметра n_years на этапе обучения определяются 
           средние приросты за год (за 2 года и т.д) для всех временных рядов 
           
           ------------------------------------------------------------------------------

       """
        self.fitGrowthFactors()

    def getPastSeriesValues(self,
                            forecasting_period: pd.Series):
        dates_for_all_previous_years = self.generateDatesForAllYears(forecasting_period,
                                                                     only_previous_years=True)
        reordered_dates = dates_for_all_previous_years.reshape(self.n_years,
                                                               len(forecasting_period)).T.reshape(-1)
        return self.series.loc[reordered_dates]

    def predictWithGrowthFactors(self, 
                                 growth_factors: np.ndarray,
                                 forecasting_period: pd.Series) -> pd.DataFrame:
        forecast = (growth_factors *
                    np.array(self.getPastSeriesValues(forecasting_period)))
        
        if self.averaging_method['forecast'] == 'geometric mean':
            forecast = np.log(forecast + 1)
        forecast = np.array(pd.DataFrame(np.concatenate((np.tile(np.arange(len(forecasting_period)).reshape(-1, 1),
                                                                 reps=(1, self.n_years)).reshape(-1, 1),
                                                         forecast),
                                                        axis=1),
                                         columns=['forec_month'] + 
                                         list(self.series.columns)).groupby('forec_month')\
                                                                   .agg(func=np.nanmean))
        forecast = (np.exp(forecast) - 1
                    if self.averaging_method['forecast'] == 'geometric mean'
                    else forecast)
        return pd.DataFrame(forecast,
                            index=forecasting_period,
                            columns=self.series.columns)

    def adjustPredictionsForDaysCounts(self, predictions: pd.DataFrame,
                                       forecasting_period: pd.Series) -> pd.DataFrame:
        if self.adjust_for_day_count:
            days_counts = self.computeDaysCounts(forecasting_period)
            predictions_ = pd.DataFrame(predictions.values *
                                        days_counts.values.reshape(-1, 1),
                                        columns=predictions.columns,
                                        index=predictions.index)
        else:
            predictions_ = predictions
        return predictions_

    def predict(self):
        """На этапе прогноза вычисляются прогнозы для всех временных рядов.
        
           --------------------------------------------------------------------------------------
        
        """
        horizons = []
        if self.horizon >= 12:
            horizons = [12] * (self.horizon // 12)
        if self.horizon % 12:
            horizons.append(self.horizon % 12)

        predictions_list, forecasting_periods = [], []

        for horizon in horizons:
            forecasting_period = pd.Series(pd.date_range(start=self.series.index[-1] + MonthEnd(1),
                                                         periods=horizon,
                                                         freq='M'))
            predictions = self.predictWithGrowthFactors(growth_factors=np.tile(self.growth_factors,
                                                                               reps=(horizon, 1)),
                                                        forecasting_period=forecasting_period)
            self.series = pd.concat((self.series,
                                     predictions),
                                    axis=0)
            predictions_list.append(predictions)
            forecasting_periods.append(forecasting_period)
        self.series = self.initial_series.copy()
        all_predictions = pd.concat(predictions_list,
                                    axis=0)
        whole_forecasting_period = pd.concat(forecasting_periods,
                                             axis=0)
        return self.adjustPredictionsForDaysCounts(all_predictions,
                                                   forecasting_period=whole_forecasting_period)

    def backtestPredict(self,
                        group: Tuple[pd.core.indexes.datetimes.DatetimeIndex,
                                     pd.DataFrame]) -> pd.DataFrame:
        latest_fact_date, series = group[0][-1], group[1]
        self.n_series_groups = len(group[0])
        self.series = pd.concat(tuple(pd.concat({str(i): series[:latest_fact_date].copy()
                                                                                  .shift(i)
                                                                                  .T},
                                                names=['shift']).T
                                      for i in range(self.n_series_groups)),
                                axis=1)
        self.fit()
        predictions = self.predict()
        if self.adjust_for_day_count:
            days_counts = self.computeDaysCounts(predictions.index)
            predictions = pd.DataFrame(predictions.values / days_counts.values.reshape(-1, 1),
                                       index=predictions.index,
                                       columns=predictions.columns)

        horizon = predictions.shape[0]

        prediction_dates = pd.concat(tuple(pd.Series(predictions.index + MonthEnd(- i))
                                           for i in range(self.n_series_groups)),
                                     axis=0)
        last_fact_dates = pd.concat(tuple(pd.Series([last_fact_date] * horizon)
                                          for last_fact_date in group[0][::-1]),
                                    axis=0)
        predictions = pd.DataFrame(np.concatenate(tuple(np.split(predictions.values,
                                                                 indices_or_sections=self.n_series_groups,
                                                                 axis=1)),
                                                  axis=0),
                                   index=prediction_dates,
                                   columns=series.columns)
        if self.adjust_for_day_count:
            predictions = pd.DataFrame(predictions.values *
                                       self.computeDaysCounts(predictions.index).values.reshape(-1, 1),
                                       index=predictions.index,
                                       columns=predictions.columns)

        predictions['last_fact_date'] = last_fact_dates.values

        predictions.index.name = date_index_name = series.index.name
        predictions = predictions.reset_index()
        predictions['lag'] = (predictions[date_index_name].dt.to_period('M') -
                              predictions['last_fact_date'].dt.to_period('M')).apply(lambda x: x.n)
        predictions = predictions.loc[predictions[date_index_name] <= series.index[-1]]
        return predictions

    def formSeriesBacktests(self):
        max_possible_backtest_dates_count = len(self.original_series.index[self.n_months + 12 * self.n_years:])
        if self.backtest_depth > max_possible_backtest_dates_count:
            series_forecast_dates = self.original_series.index[self.n_months + 12 * self.n_years:]
        else:
            series_forecast_dates = self.original_series.index[- self.backtest_depth:]
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
        self.n_series_groups = 1
        self.horizon = forecasting_horizon

        return pd.concat(predictions_list,
                         axis=0)


class SlidingSeasonalHeuristicModel(SimpleSeasonalHeuristicModel):
    """
       Класс реализует более сложную, чем в SimpleSeasonalHeuristicModel,
       сезонную эвристическую модель прогноза временного ряда.
       Расчет прогноза можно пояснить на следующем частном случае. Предположим, что прогноз
       делается на август 2023 года при условии, что последний фактический месяц -
       июнь 2023.

       прогноз на август = (средний прирост значений ряда за год *
                            фактическое значение ряда в августе 2022 +
                            средний прирост значений ряда за 2 года *
                            фактическое значение ряда в августе 2021) / 2,
       где средний прирост значений ряда за год = (факт мая 2023 / факт мая 2022 +
                                                   факт июня 2023 / факт июня 2022 +
                                                   прогноз июля 2023 / факт июля 2022) / 3,
           средний прирост значений ряда за 2 года = (факт мая 2023 / факт мая 2021 +
                                                      факт июня 2023 / факт июня 2021 +
                                                      прогноз июля 2023 / факт июля 2021) / 3.

       Здесь средние годовые приросты вычисляются отдельно для каждого прогнозного месяца. Прогноз для июля 2023 года,
       участвующий в расчете средних приростов, вычисляется ранее на основе аналогичных формул. Таким образом, прогноз
       вычисляется последовательно, вначале для июля 2023, затем для августа 2023 и т.д., т.е. как бы "скользит"
       (sliding) по прогнозным месяцам.

       Разработчик модуля Кобылкин Константин Сергеевич, ЦК ПИМОП, Сбербанк
    """
    def predictSinglePeriod(self) -> pd.DataFrame:
        forecasting_period = pd.Series(pd.date_range(start=self.series.index[-1] + MonthEnd(1),
                                                     periods=1,
                                                     freq='M'))
        return self.predictWithGrowthFactors(growth_factors=self.growth_factors,
                                             forecasting_period=forecasting_period)
    
    def fit(self):
        """
           В зависимости от значения параметра n_years на этапе обучения определяются
           средние приросты за год (за 2 года и т.д) для всех временных рядов для каждого
           прогнозного месяца в отдельности. Фактически, здесь также вычисляются и все прогнозы.

           ------------------------------------------------------------------------------

        """
        super().fit()

        for _ in range(self.horizon):
            forecasting_date = self.series.index[-1] + MonthEnd(1)
            self.series.loc[forecasting_date, :] = self.predictSinglePeriod().values.reshape(-1)
            self.fitGrowthFactors()
        self.growth_factors = None
        
    def predict(self):
        """На этапе прогноза вычисляются прогнозы для всех временных рядов.

           --------------------------------------------------------------------------------------

        """
        forecasting_period = pd.Series(self.series.iloc[-self.horizon:].index)

        predictions = self.adjustPredictionsForDaysCounts(self.series.iloc[-self.horizon:].copy(),
                                                          forecasting_period=forecasting_period)
        self.series = self.initial_series.copy()
        return predictions


class RobustSeasonalHeuristicModel(SimpleSeasonalHeuristicModel):
    """Разработчик модуля и автор идеи со сглаживанием значений - Кобылкин Константин Сергеевич, ЦК ПИМОП, Сбербанк"""
    def __init__(self,
                 past_smoothing_offset_size: int = -1,
                 future_smoothing_offset_size: int = 1,
                 smoothing_n_years: int = 2,
                 reconcile_forecasts: bool = False,
                 max_relative_values_deviations: float = 0.5,
                 growth_factors_computing_method: str = 'robust',
                 use_ratio_factors_censoring: bool = False,
                 ratio_factor_lower_threshold_quantile: float = 0.05,
                 ratio_factor_upper_threshold_quantile: float = 0.95,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.ratio_factors = None
        self.reconcile_forecasts = reconcile_forecasts
        self.past_smoothing_offset_size = past_smoothing_offset_size
        self.future_smoothing_offset_size = future_smoothing_offset_size
        self.smoothing_n_years = smoothing_n_years
        self.total_offset = (- self.past_smoothing_offset_size +
                             self.future_smoothing_offset_size + 1)
        self.max_relative_values_deviations = max_relative_values_deviations
        self.growth_factors_computing_method = growth_factors_computing_method
        self.use_ratio_factors_censoring = use_ratio_factors_censoring
        self.ratio_factor_lower_threshold_quantile = ratio_factor_lower_threshold_quantile
        self.ratio_factor_upper_threshold_quantile = ratio_factor_upper_threshold_quantile

    def getMedianValuesRatios(self):
        all_years_vals = self.getAllYearsValues()
        all_years_levels = []

        for year in range(self.n_years, -1, -1):
            if year > 0:
                year_values = all_years_vals[- self.n_months * (year + 1):
                                             - self.n_months * year, :]
                extended_year_values = all_years_vals[- self.n_months * (year + 1) - 2:
                                                      - self.n_months * year, :]
            else:
                year_values = all_years_vals[- self.n_months:, :]
                extended_year_values = all_years_vals[- self.n_months - 2:, :]

            level_deviations = (year_values[1:, :] /
                                year_values[:-1, :])
            index_of_high_level_deviations = \
                ((np.abs(level_deviations).max(axis=0) > 1 + self.max_relative_values_deviations) |
                 (np.abs(level_deviations).min(axis=0) < 1 - self.max_relative_values_deviations))

            year_levels = np.zeros(shape=(all_years_vals.shape[1], ))
            year_levels[:] = np.nan

            year_levels[index_of_high_level_deviations] = \
                np.nanmedian(extended_year_values[:, index_of_high_level_deviations],
                             axis=0)
            year_levels[~index_of_high_level_deviations] = \
                np.nanmedian(year_values[:, ~index_of_high_level_deviations],
                             axis=0)

            all_years_levels.append(year_levels.reshape(1, -1))

        years_levels = np.concatenate(all_years_levels,
                                      axis=0)

        years_levels[np.abs(years_levels) <= self.almost_zero_value] = np.nan
        return (np.tile(years_levels[-1, :].reshape(1, -1),
                        reps=(self.n_years, 1)) /
                years_levels[:-1, :])

    def generateAllDatesForSmoothing(self,
                                     dates_for_smoothing: pd.Series,
                                     usage_type: str) -> np.ndarray:
        def generateNeighboringDates(date):
            dates_before_date = pd.Series(pd.date_range(end=date,
                                                        periods=-self.past_smoothing_offset_size + 1,
                                                        freq='M'))
            dates_after_date = pd.Series(pd.date_range(start=date + MonthEnd(1),
                                                       periods=self.future_smoothing_offset_size,
                                                       freq='M'))
            return pd.concat((dates_before_date,
                              dates_after_date),
                             axis=0).reset_index(drop=True)

        neighboring_dates = dates_for_smoothing.apply(generateNeighboringDates)
        if usage_type == 'ratio_factors':
            neighboring_dates_over_years = tuple(pd.DataFrame(neighboring_dates).reset_index(drop=True) + MonthEnd(i * 12)
                                                 for i in range(- self.smoothing_n_years + 1, 1))
            return np.array(pd.concat(neighboring_dates_over_years,
                                      axis=0)).reshape(-1)
        elif usage_type == 'past_series_values':
            return np.array(neighboring_dates).reshape(-1)

    def addExtraDatesForSmoothing(self) -> pd.Series:
        dates_before_series_dates = pd.Series(pd.date_range(end=self.series.index[0] + MonthEnd(-1),
                                                            periods=(-self.past_smoothing_offset_size +
                                                                     12 * (self.smoothing_n_years - 1)),
                                                            freq='M'),
                                              index=range(self.past_smoothing_offset_size -
                                                          12 * (self.smoothing_n_years - 1), 0))
        dates_after_series_dates = pd.Series(pd.date_range(start=self.series.index[-1] + MonthEnd(1),
                                                           periods=self.future_smoothing_offset_size,
                                                           freq='M'),
                                             index=range(len(self.series),
                                                         len(self.series) +
                                                         self.future_smoothing_offset_size))
        extended_series_dates = pd.concat((dates_before_series_dates,
                                           pd.Series(self.series.index),
                                           dates_after_series_dates),
                                          axis=0)
        extended_series_dates.name = 'Date'
        return extended_series_dates

    def getRawRatioFactors(self,
                           augmented_series: pd.DataFrame,
                           all_dates_for_smoothing: np.ndarray):
        all_vals_for_smoothing = np.array(augmented_series.loc[all_dates_for_smoothing])\
                                   .reshape(-1, self.total_offset,
                                            self.series.shape[1])
        all_vals_for_smoothing[~np.isnan(all_vals_for_smoothing) &
                               (np.abs(all_vals_for_smoothing) <= self.almost_zero_value)] = np.nan
        return (all_vals_for_smoothing[:, [-self.past_smoothing_offset_size], :] /
                all_vals_for_smoothing)

    def getAveragedRatioFactors(self,
                                raw_ratio_factors: np.ndarray):
        horizon = raw_ratio_factors.shape[0] // (self.n_years * self.smoothing_n_years)
        months = np.tile(np.tile(np.arange(horizon),
                                 reps=self.n_years),
                         reps=self.smoothing_n_years)
        years = np.tile(np.tile(np.arange(-self.n_years, 0, 1).reshape(-1, 1),
                                reps=(1, horizon)).reshape(-1),
                        reps=self.smoothing_n_years)
        grouping_cols = np.concatenate((months.reshape(-1, 1),
                                        years.reshape(-1, 1)),
                                       axis=1)
        sorted_grouping_cols_index = np.lexsort((grouping_cols[:, 0],
                                                 grouping_cols[:, 1]),
                                                axis=0)
        raw_ratio_factors = raw_ratio_factors[sorted_grouping_cols_index, ...]

        n_splits = grouping_cols.shape[0] // self.smoothing_n_years
        splits = np.split(raw_ratio_factors,
                          indices_or_sections=n_splits)
        grouped_ratio_factors = np.concatenate(tuple(np.expand_dims(split,
                                                                    axis=1)
                                                     for split in splits),
                                               axis=1)
        grouped_ratio_factors = np.exp(np.nanmean(np.log(grouped_ratio_factors),
                                                  axis=0))
        return grouped_ratio_factors

    def computeRatioFactors(self,
                            dates_for_all_previous_years: pd.Series,
                            augmented_series: pd.DataFrame) -> np.ndarray:
        all_dates_for_ratio_factors = self.generateAllDatesForSmoothing(dates_for_all_previous_years,
                                                                        usage_type='ratio_factors')
        self.ratio_factors = self.getAveragedRatioFactors(
            raw_ratio_factors=self.getRawRatioFactors(augmented_series,
                                                      all_dates_for_ratio_factors))
        if self.use_ratio_factors_censoring:
            self.ratio_factors = self.censorFactors(
                factors=self.ratio_factors,
                factor_lower_threshold_quantile=self.ratio_factor_lower_threshold_quantile,
                factor_upper_threshold_quantile=self.ratio_factor_upper_threshold_quantile
            )

    def extractRawPastSeriesValues(self,
                                   dates_for_all_previous_years: pd.Series,
                                   augmented_series: pd.DataFrame) -> np.ndarray:
        all_dates_for_past_series_values = self.generateAllDatesForSmoothing(
            dates_for_all_previous_years,
            usage_type='past_series_values'
        )
        all_past_series_values = np.array(augmented_series.loc[all_dates_for_past_series_values]) \
            .reshape(-1, self.total_offset,
                     self.series.shape[1])

        all_past_series_values[~np.isnan(all_past_series_values) &
                               (np.abs(all_past_series_values) <= self.almost_zero_value)] = np.nan
        return all_past_series_values

    def getPastSeriesValues(self,
                            forecasting_period: pd.Series):
        dates_for_all_previous_years = pd.Series(self.generateDatesForAllYears(forecasting_period,
                                                                               only_previous_years=True))
        augmented_series = pd.concat((self.series.reset_index(drop=True),
                                      self.addExtraDatesForSmoothing()),
                                     axis=1).set_index('Date').sort_index()
        self.computeRatioFactors(dates_for_all_previous_years,
                                 augmented_series)
        all_past_series_values = self.extractRawPastSeriesValues(dates_for_all_previous_years,
                                                                 augmented_series)

        smoothed_series = pd.DataFrame(np.nanmedian(self.ratio_factors *
                                                    all_past_series_values,
                                                    axis=1),
                                       index=dates_for_all_previous_years,
                                       columns=self.series.columns)

        reordered_dates = np.array(dates_for_all_previous_years).reshape(self.n_years,
                                                                         len(forecasting_period))\
                                                                .T.reshape(-1)
        return smoothed_series.loc[reordered_dates]

    def fitGrowthFactors(self):
        if self.growth_factors_computing_method == 'classical':
            super().fitGrowthFactors()
        elif self.growth_factors_computing_method == 'robust':
            self.growth_factors = self.getMedianValuesRatios()
        if self.use_growth_factor_censoring:
            self.growth_factors = self.censorFactors(
                factors=self.growth_factors,
                factor_lower_threshold_quantile=self.growth_factor_lower_threshold_quantile,
                factor_upper_threshold_quantile=self.growth_factor_upper_threshold_quantile
            )

    def reconcileForecasts(self,
                           single_forecasting_period: pd.Series,
                           single_forecasting_period_forecasts: pd.DataFrame,
                           single_forecasting_period_id: int):
        latest_dates_for_smoothing = self.generateAllDatesForSmoothing(
            single_forecasting_period,
            usage_type='past_series_values')
        initial_series = self.series.copy()
        self.series = pd.concat((self.series,
                                 single_forecasting_period_forecasts),
                                axis=0)
        augmented_series = pd.concat((self.series.reset_index(drop=True),
                                      self.addExtraDatesForSmoothing()),
                                     axis=1).set_index('Date').sort_index()

        latest_series_values = np.array(augmented_series.loc[latest_dates_for_smoothing]) \
            .reshape(1, self.total_offset,
                     self.series.shape[1])
        latest_series_values[~np.isnan(latest_series_values) &
                             (np.abs(latest_series_values) <= self.almost_zero_value)] = np.nan
        ratio_factors = self.ratio_factors[- self.horizon +
                                           single_forecasting_period_id, ...].reshape(1, self.total_offset,
                                                                                      self.series.shape[1])
        self.series = initial_series
        return np.nanmedian(ratio_factors * latest_series_values,
                            axis=1)

    def predictWithGrowthFactors(self,
                                 growth_factors: np.ndarray,
                                 forecasting_period: pd.Series) -> pd.DataFrame:
        forecasts = super().predictWithGrowthFactors(
            growth_factors=growth_factors,
            forecasting_period=forecasting_period)

        if self.reconcile_forecasts:
            reconciled_forecasts = []
            for date_id, date in enumerate(forecasting_period):
                single_forecasting_period_forecasts = pd.DataFrame(forecasts.loc[date].values.reshape(1, -1),
                                                                   columns=self.series.columns,
                                                                   index=[date])
                single_forecasting_period_reconciled_forecasts = \
                    self.reconcileForecasts(single_forecasting_period=pd.Series([date]),
                                            single_forecasting_period_forecasts=single_forecasting_period_forecasts,
                                            single_forecasting_period_id=date_id)
                single_period_reconciled_forecasts = pd.DataFrame(
                    single_forecasting_period_reconciled_forecasts.reshape(1, -1),
                    columns=self.series.columns,
                    index=[date]
                )
                self.series = pd.concat((self.series,
                                         single_period_reconciled_forecasts),
                                        axis=0)
                reconciled_forecasts.append(single_period_reconciled_forecasts)
            forecasts = pd.concat(reconciled_forecasts,
                                  axis=0)
        return forecasts




