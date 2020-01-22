from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import style
style.use('dark_background')

from itertools import chain

import pandas as pd
import numpy as np

from sklearn.cluster import MeanShift, estimate_bandwidth


def get_levels(data_frame, go_back=32):
    levels = list()

    reindex = data_frame.set_index(pd.DatetimeIndex(data_frame.Date))
    logic = {'Open': 'first',
             'High': 'max',
             'Low': 'min',
             'Close': 'last',
             'Volume': 'sum'}

    offset = pd.offsets.timedelta(days=-6)
    # f = pd.read_clipboard(parse_dates=['Date'], index_col=['Date'])
    data_frame = reindex.resample('W', loffset=offset).apply(logic).reset_index()

    if len(data_frame) > go_back:
        for i in range(1):
            X = np.sort(np.array(pd.concat(
                [data_frame.High[len(data_frame) - go_back * (i + 1):len(data_frame) - go_back * (i)],
                 data_frame.Low[len(data_frame) - go_back * (i + 1):len(data_frame) - go_back * (i)]]).reset_index(
                drop=True)))
            X = np.reshape(X, (-1, 1))

            bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(X)
            levels.append(ms.cluster_centers_)
    else:
        X = np.sort(np.array(pd.concat([data_frame.High, data_frame.Low]).reset_index(drop=True)))
        X = np.reshape(X, (-1, 1))

        bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        levels.append(ms.cluster_centers_)

    return np.array(list(chain(*levels)))


def get_pivots(data_frame, proc=False):
    """
    camarilla pivots
    """
    reindex = data_frame.set_index(pd.DatetimeIndex(data_frame.Date))
    logic = {'Open': 'first',
             'High': 'max',
             'Low': 'min',
             'Close': 'last',
             'Volume': 'sum'}

    # offset = pd.offsets.timedelta(days=-30)
    # f = pd.read_clipboard(parse_dates=['Date'], index_col=['Date'])
    data_frame = reindex.resample('M').apply(logic).reset_index()

    H = data_frame.High
    C = data_frame.Close
    L = data_frame.Low

    if not proc:
        H = H.iloc[-1]
        C = C.iloc[-1]
        L = L.iloc[-1]

    p = pd.Series((H + L + C) / 3).rename('p')

    S1 = pd.Series(C - (H - L) * 1.1 / 12).rename('S1')
    S2 = pd.Series(C - (H - L) * 1.1 / 6).rename('S2')
    S3 = pd.Series(C - (H - L) * 1.1 / 4).rename('S3')
    S4 = pd.Series(C - (H - L) * 1.1 / 2).rename('S4')

    R1 = pd.Series(C + (H - L) * 1.1 / 12).rename('R1')
    R3 = pd.Series(C + (H - L) * 1.1 / 4).rename('R2')
    R2 = pd.Series(C + (H - L) * 1.1 / 6).rename('R3')
    R4 = pd.Series(C + (H - L) * 1.1 / 2).rename('R4')

    return (pd.DataFrame([R4, R3, R2, R1, p, S1, S2, S3, S4]).transpose()).join(data_frame.Date)


def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n + 1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100. / (1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n

        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi


def draw_vis(input_data, show_levels=False):
    # Creating required data in new DataFrame OHLC
    data_for_candles = input_data.copy()
    data_for_candles["Date"] = data_for_candles["Date"].apply(mdates.date2num)
    ohlc = data_for_candles[['Date', 'Open', 'High', 'Low', 'Close']].copy()

    fig = plt.figure(figsize=(10, 5))
    fig.canvas.set_window_title('Support And Resistance')

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    ax.xaxis.set_visible(False)
    candlestick_ohlc(ax, ohlc.values, colorup='green', colordown='red')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    #     ax.grid(True)

    if show_levels:
        levels = get_levels(input_data)
        ax3 = fig.add_subplot(gs[0])
        ax3.xaxis.set_visible(False)
        for lvl in levels:
            ax3.plot(data_for_candles["Date"], np.repeat(lvl, len(data_for_candles["Date"])), 'w')

    ax5 = fig.add_subplot(gs[1], sharex=ax)
    rsi = rsiFunc(data_for_candles.Close)
    ax5.plot(data_for_candles['Date'], rsi, color='b')
    ax5.axhline(30, color='r')
    ax5.axhline(70, color='r')
    ax5.axhline(50, color='g')

    plt.tight_layout()
    plt.show()


candle_chart = pd.read_pickle('btc_data.pkl')
last_n_days = 180
draw_vis(candle_chart[-last_n_days:], True)