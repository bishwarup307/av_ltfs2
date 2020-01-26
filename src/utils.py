'''
 Created on Mon Jan 20 2020
 __author__: bishwarup
'''
import os
import sys
import pandas as pd
import numpy as np
import copy
import time
import re
from fbprophet import Prophet
from collections import defaultdict
from datetime import datetime
from functools import partial
import warnings
from tqdm import tqdm_notebook as tqdm
from multiprocessing import Pool
from difflib import get_close_matches
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from jupyterthemes import jtplot
import requests
from bs4 import BeautifulSoup
import seaborn as sns
from matplotlib.dates import (
        MonthLocator,
        num2date,
        AutoDateLocator,
        AutoDateFormatter,
    )
from fbprophet.diagnostics import (
        cross_validation, 
        performance_metrics
)
from fbprophet.plot import plot_cross_validation_metric

from config import BASE_DIR, BASE_URL, statewise_params, holiday_window

def plotProphet(
    m, fcst, ax=None, uncertainty=True, plot_cap=True, xlabel='ds', ylabel='y', figsize=(22, 7)):
    """Plot the Prophet forecast.
    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    ax: Optional matplotlib axes on which to plot.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    xlabel: Optional label name on X-axis
    ylabel: Optional label name on Y-axis
    figsize: Optional tuple width, height in inches.
    Returns
    -------
    A matplotlib figure.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    fcst_t = fcst['ds'].dt.to_pydatetime()
    ax.plot(m.history['ds'].dt.to_pydatetime(), m.history['y'], 'k.', color = 'white')
    ax.plot(fcst_t, fcst['yhat'], ls='-', c='#0072B2')
    if 'cap' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['cap'], ls='--', c='k')
    if m.logistic_floor and 'floor' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['floor'], ls='--', c='k')
    if uncertainty and m.uncertainty_samples:
        ax.fill_between(fcst_t, fcst['yhat_lower'], fcst['yhat_upper'],
                        color='#0072B2', alpha=0.2)
    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig

def MAPE(a, f):
    return 100. * np.sum(np.abs(1. - f / a)) / len(a)

def sunday(ds):
    date = pd.to_datetime(ds)
    if date.weekday() == 6: return 1
    else: return 0

def scrape_national_holidays(years = [2018, 2019]):
    holiday_df = []
    for year in years:
        url = f'https://www.mapsofindia.com/events/{year}-holidays-calendar.html'
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'lxml')
        table = soup.find('table', {'class':'tableizer-table'})
        rows = table.find_all('tr')
        headers = rows[0].find_all('th')
        if len(headers) > 0: headers = [h.get_text() for h in headers]
        df = []
        for row in rows[1:]:
            cols = row.find_all('td')
            record = [col.get_text() for col in cols]
            df.append(record)
        df = pd.DataFrame(df, columns=headers)
        df.columns = [x.lower().replace(" ", "_") for x in df.columns]
        df = df.query('holiday_type == "Gazetted Holiday"')
        holiday_df.append(df)
    holiday_df = pd.concat(holiday_df)
    holiday_df["date"] = holiday_df["date"].map(lambda x: datetime.strptime(x, "%b %d, %Y").strftime("%Y-%m-%d"))
    holiday_df["date"] = pd.to_datetime(holiday_df["date"])
    holiday_df["date"][holiday_df["holiday_name"] == "Muharram/Ashura"] = holiday_df["date"] - pd.Timedelta(11, unit='days')
    holiday_df = holiday_df[["date", "holiday_name"]]
    holiday_df.rename(columns = {"date": "ds", "holiday_name": "holiday"}, inplace = True)
    holiday_df["lower_window"] = holiday_df["holiday"].map(lambda x: holiday_window.get(x, (0, 0))[0])
    holiday_df["upper_window"] = holiday_df["holiday"].map(lambda x: holiday_window.get(x, (0, 0))[1])
    return holiday_df

def format_holidays(holiday_list, match_cutoff = 0.8):
    matched = dict(zip(set(holiday_list), np.repeat(False, len(set(holiday_list)))))
    holiday_mapper = defaultdict(str)
    for holiday in set(holiday_list):
        if not matched[holiday]:
            matches = get_close_matches(holiday, set(holiday_list), cutoff = match_cutoff)
#             if len(set(matches)) == 1:
#                 matched[holiday] = True
#            else:
            td = {k: matches[0] for k in matches}
            holiday_mapper.update(td)
            for item in matches:
                matched[item] = True
    return holiday_mapper

def scrape_holidays(state, year, exclude = ['Day', 'Comments']):
    state = state.lower().replace(' ', '-')
    exclude_indices = []
    url = f'{BASE_URL}/{state}/{year}'
    content = requests.get(url)
    if content.status_code == 404:
        raise ValueError(f"Couldn't find state: {state}")
    page = content.text
    soup = BeautifulSoup(page, 'lxml')
    table = soup.find('table', {'class':'country-table'})
    
    rows = table.find_all('tr')
    headers = rows[0].find_all('th')
    if len(headers) > 0: headers = [h.get_text() for h in headers]
    if len(exclude) > 0: 
        exclude_indices = [headers.index(item) for item in exclude]
        for item in exclude:
            headers.remove(item)
    #return headers
    df = []
    for row in rows[1:]:
        cols = row.find_all('td')
        #assert len(cols) == len(headers)
        records = []
        for i, col in enumerate(cols):
            record = col.find('time')['datetime'] if i == 1 else col.get_text()  # need to fetch the date from the attribute value
            if i not in exclude_indices: records.append(record)
        df.append(records)
    df = pd.DataFrame(df, columns=headers)
    df['state'] = state
    df.rename(columns = {'Holiday Name': 'holiday', 'Date': 'ds'}, inplace = True)
    return df

def get_holidays(state, years= [2017, 2018, 2019], exclude = ['Day', 'Comments'], match_cutoff = 0.8, 
                 keep_holidays = ['Regional Holiday', 'Public Holiday'], save = os.path.join(BASE_DIR, "input")):
    if state == 'ORISSA': state = 'Odisha'
    if save is not None:
        try:
            holiday_df = pd.read_csv(os.path.join(save, f'{state}.csv'), parse_dates = ['ds'])
            return holiday_df
        except IOError:
            pass
    try:
        holiday_df = []
        for year in years:
            d = scrape_holidays(state, year, exclude=exclude)
            holiday_df.append(d)
        holiday_df = pd.concat(holiday_df)
    except:
        warnings.warn('holiday list download failed, will use inbuilt holidays!')
        holiday_df = None
    
    if holiday_df is not None:
        if keep_holidays is not None: holiday_df = holiday_df[holiday_df.Type.isin(keep_holidays)]
        holiday_df['ds'] = pd.to_datetime(holiday_df['ds'])
        holiday_mapper = format_holidays(holiday_df.holiday.tolist())
        #if state == 'karnataka' : holiday_mapper.update({'Ugadi': 'Chandramana Ugadi'})
        holiday_df['holiday'] = holiday_df['holiday'].map(holiday_mapper)
        cnt = holiday_df.holiday.value_counts()
        keep_holidays = cnt[cnt == 3].index
        holiday_df = holiday_df[holiday_df.holiday.isin(keep_holidays)]
        holiday_df.drop(['Type', 'state'], axis = 1, inplace = True)
        
    if (holiday_df is not None) and (save is not None): holiday_df.to_csv(os.path.join(save, f'{state}.csv')) 
    return holiday_df

def build_model(params, state, verbose = True):    
    if params['holiday']:
        holiday_df = get_holidays(state = state)
        if holiday_df is not None:
            if verbose: print('initializing model with downloaded holidays!')
            m = Prophet(daily_seasonality=False, 
                        weekly_seasonality=False, 
                        yearly_seasonality=False, 
                        changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
                        holidays=holiday_df, 
                        holidays_prior_scale=params.get('holidays_prior_scale', 10),
                        interval_width=0.95)
        else:
            if verbose: print('using in-built holidays!')
            m = Prophet(daily_seasonality=False, 
                        weekly_seasonality=False, 
                        yearly_seasonality=False, 
                        changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
                        holidays_prior_scale=params.get('holidays_prior_scale', 10),
                        interval_width=0.95)
            m.add_country_holidays(country_name='IN')
    else:
        m = Prophet(
            daily_seasonality=False, 
            weekly_seasonality=False, 
            yearly_seasonality=False,
            changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
            interval_width=0.95
        )
    
    if 'seasonal' in params.keys():
        if 'yearly' in params['seasonal'].keys():
            if verbose: print('using yearly seasonality')
            period, fourier_order, prior_scale, mode = params['seasonal']['yearly']
            m.add_seasonality(name = 'yearly', period = period, fourier_order = fourier_order, prior_scale = prior_scale, mode = mode)
        if 'quarterly' in params['seasonal'].keys():
            if verbose: print('using quarterly seasonality')
            period, fourier_order, prior_scale, mode = params['seasonal']['quarterly']
            m.add_seasonality(name = 'quarterly', period = period, fourier_order = fourier_order, prior_scale = prior_scale, mode = mode)
        if 'monthly' in params['seasonal'].keys():
            if verbose: print('using monthly seasonality')
            period, fourier_order, prior_scale, mode = params['seasonal']['monthly']
            m.add_seasonality(name = 'monthly', period = period, fourier_order = fourier_order, prior_scale = prior_scale, mode = mode)
        if 'weekly' in params['seasonal'].keys():
            if verbose: print('using weekly seasonality')
            period, fourier_order, prior_scale, mode = params['seasonal']['weekly']
            m.add_seasonality(name = 'weekly', period = period, fourier_order = fourier_order, prior_scale = prior_scale, mode= mode)
        if 'daily' in params['seasonal'].keys():
            if verbose: print('using daily seasonality')
            period, fourier_order, prior_scale, mode = params['seasonal']['daily']
            m.add_seasonality(name = 'daily', period = period, fourier_order = fourier_order, prior_scale = prior_scale, mode= mode)
    
    return m

def build_aggregated_model(params, holiday_df = None, verbose = True):    
    if params['holiday']:
        if holiday_df is not None:
            m = Prophet(daily_seasonality=False, 
                        weekly_seasonality=False, 
                        yearly_seasonality=False, 
                        holidays = holiday_df,
                        changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
                        holidays_prior_scale=params.get('holidays_prior_scale', 10),
                        interval_width=0.95)
        else:
            m = Prophet(daily_seasonality=False, 
                        weekly_seasonality=False, 
                        yearly_seasonality=False, 
                        changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
                        holidays_prior_scale=params.get('holidays_prior_scale', 10),
                        interval_width=0.95)
            m.add_country_holidays(country_name='IN')
    else:
        m = Prophet(
            daily_seasonality=False, 
            weekly_seasonality=False, 
            yearly_seasonality=False,
            changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
            interval_width=0.95
        )
    
    if 'seasonal' in params.keys():
        if 'yearly' in params['seasonal'].keys():
            if verbose: print('using yearly seasonality')
            period, fourier_order, prior_scale, mode = params['seasonal']['yearly']
            m.add_seasonality(name = 'yearly', period = period, fourier_order = fourier_order, prior_scale = prior_scale, mode = mode)
        if 'quarterly' in params['seasonal'].keys():
            if verbose: print('using quarterly seasonality')
            period, fourier_order, prior_scale, mode = params['seasonal']['quarterly']
            m.add_seasonality(name = 'quarterly', period = period, fourier_order = fourier_order, prior_scale = prior_scale, mode = mode)
        if 'monthly' in params['seasonal'].keys():
            if verbose: print('using monthly seasonality')
            period, fourier_order, prior_scale, mode = params['seasonal']['monthly']
            m.add_seasonality(name = 'monthly', period = period, fourier_order = fourier_order, prior_scale = prior_scale, mode = mode)
        if 'weekly' in params['seasonal'].keys():
            if verbose: print('using weekly seasonality')
            period, fourier_order, prior_scale, mode = params['seasonal']['weekly']
            m.add_seasonality(name = 'weekly', period = period, fourier_order = fourier_order, prior_scale = prior_scale, mode= mode)
        if 'daily' in params['seasonal'].keys():
            if verbose: print('using daily seasonality')
            period, fourier_order, prior_scale, mode = params['seasonal']['daily']
            m.add_seasonality(name = 'daily', period = period, fourier_order = fourier_order, prior_scale = prior_scale, mode= mode)
    
    return m

def validate_segment2(df, state, params, validate = 60, apply_log = False, use_best_params = False, 
                    cutoff_date = None, low_cap = 0, add_sunday = 0, verbose = True):
    fit_df = df.query('state == @state & segment == 2')
    fit_df = fit_df[["application_date", "case_count"]].rename(
        columns = {'application_date' : 'ds', 'case_count': 'y'}
        )
    fit_df.sort_values('ds', inplace= True)
    if apply_log: fit_df['y'] = np.log1p(fit_df['y'])

    if add_sunday > 0: fit_df['sunday'] = fit_df['ds'].map(sunday)

    if cutoff_date is not None:
        fit_train = fit_df.query('ds < @cutoff_date')
        fit_valid = fit_df.query('ds >= @cutoff_date')[:90]
    else:
        fit_train = fit_df.iloc[:-validate]
        fit_valid = fit_df.iloc[-validate:]
    #print(fit_valid.shape)
    
    m = build_model(statewise_params[state], state, verbose=verbose) if use_best_params else build_model(params, state, verbose=verbose)
    if add_sunday > 0: m.add_regressor('sunday', prior_scale = add_sunday)
    m.fit(fit_train)

    forecast = m.predict(fit_valid)
    fc = forecast[["ds", "yhat"]]
    fc = fc.merge(fit_valid, on = 'ds', how = 'left')
    if apply_log:
        fc['y'] = np.exp(fc.y) - 1
        fc['yhat'] = np.exp(fc.yhat) - 1
    fc['yhat'] = np.clip(fc.yhat.values, low_cap, fc.yhat.max())
    mape = MAPE(fc.y.values[fc.y > 0], fc.yhat.values[fc.y > 0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
                    x=fc.ds,
                    y=fc['y'],
                    name="Actual",
                    line_color='deepskyblue',
                    mode='lines+markers',
                    opacity=0.8))

    fig.add_trace(go.Scatter(
                    x=fc.ds,
                    y=fc['yhat'],
                    name="Forecast",
                    line = dict(color='firebrick', width=4, dash='dot'),
                    opacity=0.8))

    # fig.add_trace(go.Scatter(
    #     x = fc.ds,
    #     y = fc['yhat2'],
    #     name = 'Forecast Future',
    #     line = dict(color='limegreen', width=3, dash='dash'),
    #     opacity=0.8
    # ))

    fig.update_layout(
                        xaxis_range=[fc.ds.min(), fc.ds.max()],
                        title_text=f"MAPE: {mape:.2f}", 
                        template = 'plotly_dark'
                    )
    fig.show()

def validate_segment1(df, validate, branch_id):
    #print("here")
    tdf = df.query('branch_id == @branch_id & segment == 1')[['application_date', 'case_count']].\
                rename(columns = {'application_date' : 'ds', 'case_count': 'y'}).copy()
    tdf.sort_values('ds', inplace = True)

    df_train = tdf.iloc[:-validate].copy()
    df_valid = tdf.iloc[-validate:].copy()
    m = Prophet(daily_seasonality = False)
    #m.add_seasonality(name = 'yearly', period = 365, fourier_order = 1, prior_scale = 0.05)
    #m.add_seasonality(name = 'quarterly', period = 91, fourier_order = 10)
    #m.add_seasonality(name = 'monthly', period = 30, fourier_order = 1, prior_scale = 0.05)
    #m.add_seasonality(name = 'weekly', period = 7, fourier_order = 1)
    m.add_country_holidays(country_name='IN')
    m.fit(df_train)
    fc = m.predict(df_valid)['yhat'].values.tolist()
    return list(zip(df_valid['ds'].astype(str).tolist(), df_valid['y'].tolist(), fc))
    
def imap_unordered_bar(func, args, n_processes = 12):
    p = Pool(n_processes)
    res_list = []
    with tqdm(total = len(args)) as pbar:
        for _, res in enumerate(p.imap_unordered(func, args)):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list

def segment2(df, validate, state):
    tdf = df.query('state == @state & segment == 2')[['application_date', 'case_count']].\
                rename(columns = {'application_date' : 'ds', 'case_count': 'y'}).copy()
    tdf.sort_values('ds', inplace = True)

    df_train = tdf.iloc[:-validate].copy()
    df_valid = tdf.iloc[-validate:].copy()
    params = statewise_params.get(state, None)
    m = Prophet() if params is None else build_model(params, state, verbose=False)
    m.fit(df_train)
    fc = m.predict(df_valid)['yhat'].values.tolist()
    return list(zip(df_valid['ds'].astype(str).tolist(), df_valid['y'].tolist(), fc))

def cv(df, horizon = 60):
    print('*' * 25)
    print('validating segment 1')
    print('*' * 25)
    branch_ids = df['branch_id'].dropna().astype(np.int32).unique()
    segment1 = partial(validate_segment1, df, horizon)
    result = imap_unordered_bar(segment1, args = branch_ids)
    gt = defaultdict(float)
    forecasted = defaultdict(float)
    
    for _, pred in tqdm(enumerate(result)):
        for tpl in pred:
            dt, y, yhat = tpl
            yhat = np.max([yhat, 0])
            gt[dt] = gt[dt] + y
            forecasted[dt] = forecasted[dt] + yhat

    gt = pd.DataFrame.from_dict(gt, orient='index').reset_index().rename(columns = {'index' : 'ds', 0: 'y'})
    forecasted = pd.DataFrame.from_dict(forecasted, orient='index').reset_index().rename(columns = {'index' : 'ds', 0: 'yhat'})
    D = gt.merge(forecasted, on = 'ds', how = 'left')
    D['ds'] = pd.to_datetime(D['ds'])
    D.sort_values('ds', inplace = True)
    valid = D.iloc[:-1]
    mape1 = MAPE(valid.y.values, valid.yhat.values)

    print('*' * 25)
    print('validating segment 2')
    print('*' * 25)
    states = df.query('segment == 2').state.dropna().unique()
    s2 = partial(segment2, df, horizon)
    result = imap_unordered_bar(s2, args = states)
    gt = defaultdict(float)
    forecasted = defaultdict(float)
    for _, pred in tqdm(enumerate(result)):
        for tpl in pred:
            dt, y, yhat = tpl
            yhat = np.max([yhat, 0])
            gt[dt] = gt[dt] + y
            forecasted[dt] = forecasted[dt] + yhat
    gt = pd.DataFrame.from_dict(gt, orient='index').reset_index().rename(columns = {'index' : 'ds', 0: 'y'})
    forecasted = pd.DataFrame.from_dict(forecasted, orient='index').reset_index().rename(columns = {'index' : 'ds', 0: 'yhat'})
    D = gt.merge(forecasted, on = 'ds', how = 'left')
    D['ds'] = pd.to_datetime(D['ds'])
    D.sort_values('ds', inplace = True)
    mape2 = MAPE(D.y.values, D.yhat.values)

    print(f'segment 1 MAPE: {mape1:.2f}')
    print(f'segment 2 MAPE: {mape2:.2f}')
    print(f'cv: {0.5 * mape1 + 0.5 * mape2 :.2f}')
    #print(MAPE(valid.y.values, valid.yhat.values))

def predict_segment2(train, test, state):
    tdf = train.query('state == @state & segment == 2')[['application_date', 'case_count']].\
                rename(columns = {'application_date' : 'ds', 'case_count': 'y'}).copy()
    tdf.sort_values('ds', inplace = True)

    params = statewise_params.get(state, None)
    m = Prophet() if params is None else build_model(params, state, verbose=False)
    m.fit(tdf)
    fc = m.predict(test)['yhat'].values.tolist()
    return list(zip(test['ds'].astype(str).tolist(), fc))

def predict_segment1(train, test, branch_id):
    tdf = train.query('branch_id == @branch_id & segment == 1')[['application_date', 'case_count']].\
                rename(columns = {'application_date' : 'ds', 'case_count': 'y'}).copy()
    tdf.sort_values('ds', inplace = True)
    m = Prophet(daily_seasonality = False)
    m.add_country_holidays(country_name='IN')
    m.fit(tdf)
    fc = m.predict(test)['yhat'].values.tolist()
    return list(zip(test['ds'].astype(str).tolist(), fc))

def fit_predict(train, test):
    branch_ids = train['branch_id'].dropna().astype(np.int32).unique()
    ts1 = test.query('segment == 1')[['application_date']].rename(columns = {'application_date': 'ds'})
    segment1 = partial(predict_segment1, train, ts1)
    result = imap_unordered_bar(segment1, args = branch_ids)
    forecasted = defaultdict(float)
    for _, pred in tqdm(enumerate(result)):
        for tpl in pred:
            dt, yhat = tpl
            yhat = np.max([yhat, 0]) # to be checked
            forecasted[dt] = forecasted[dt] + yhat
    pred_segment1 = pd.DataFrame.from_dict(forecasted, orient='index').reset_index().rename(columns = 
                    {'index' : 'application_date', 0: 'case_count'})
    pred_segment1['segment'] = 1

    ts2 = test.query('segment == 2')[['application_date']].rename(columns = {'application_date': 'ds'})
    states = train.query('segment == 2').state.dropna().unique()
    s2 = partial(predict_segment2, train, ts2)
    result = imap_unordered_bar(s2, args = states)
    forecasted = defaultdict(float)
    for _, pred in tqdm(enumerate(result)):
        for tpl in pred:
            dt, yhat = tpl
            yhat = np.max([yhat, 0])
            forecasted[dt] = forecasted[dt] + yhat
    pred_segment2 = pd.DataFrame.from_dict(forecasted, orient='index').reset_index().rename(columns = 
                {'index' : 'application_date', 0: 'case_count'})
    pred_segment2['segment'] = 2
    
    pred = pd.concat([pred_segment1, pred_segment2], ignore_index = True)
    return pred