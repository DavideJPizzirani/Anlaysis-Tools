import pandas as pd
import numpy as np
from pandas import DataFrame
#import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
from matplotlib.ticker import AutoLocator, AutoMinorLocator
from scipy import signal
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import iplot
import plotly.express as px
import seaborn as sns

red = '\033[91m'
green = '\033[92m'
blue = '\033[94m'
bold = '\033[1m'
italics = '\033[3m'
underline = '\033[4m'
end = '\033[0m'

def flip_df_x(df):
    df.x = df.x * -1
    return df

def flip_df_y(df):
    df.y = df.y * -1
    return df

def flip_multiple_df_x(df):
    for df in df:
        df.x = df.x * -1
    return df

def flip_multiple_df_y(df):
    for df in df:
        df.y = df.y * -1
    return df

def symmetrize_list(lis, temperature_thickness, name, show = True, save = True):
    assert len(lis) == 2
    assert type(name) is str
    df_symm = lis[1].copy()
    df_symm.y = 0.5*(df_symm.y + np.flip(lis[0].y.to_numpy()))
    if show is True:
        dfs = {"Symmetrized "+str(temperature_thickness): df_symm}
        fig = go.Figure()
        fig.update_layout(
        title="Symmetrized longitudinal resistivity",
        xaxis_title= "\u03BC\u2080H (T)",
        yaxis_title="\u03C1\u02E3\u02E3 (\u03BC\u03A9 cm)",
        legend_title="Signal",
        font=dict(
            family="Times New Roman, monospace",
            size=18,
            color="black"))
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
        for i in dfs:
            fig = fig.add_trace(go.Scatter(x = dfs[i].x, y = dfs[i].y, name = i, mode = 'markers'))
        fig.show()
    if save is True:
        df_symm.to_csv(name + '_symmetrized.csv', sep='\t', encoding='utf-8')
    return df_symm


def asymmetrize_list(lis, temperature_thickness, name, show = True, save = True):
    assert len(lis) == 2
    assert type(name) is str
    df_asymm = lis[1].copy()
    df_asymm.y = 0.5*(df_asymm.y - np.flip(lis[0].y.to_numpy()))
    if show is True:
        dfs = {"Anti-symmetrized "+str(temperature_thickness): df_asymm}
        fig = go.Figure()
        fig.update_layout(
        title="Anti-symmetrized transverse resistivity",
        xaxis_title= "\u03BC\u2080H (T)",
        yaxis_title="\u03C1\u02B8\u02E3 (\u03BC\u03A9 cm)",
        legend_title="Signal",
        font=dict(
            family="Times New Roman, monospace",
            size=18,
            color="black"))
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
        for i in dfs:
            fig = fig.add_trace(go.Scatter(x = dfs[i].x, y = dfs[i].y, name = i, mode = 'markers'))
        fig.show()
    if save is True:
        df_asymm.to_csv(name + '_anti_symmetrized.csv', sep='\t', encoding='utf-8')
    return df_asymm


def symmetrize_list_noflip(lis, temperature_thickness, name, show = True, save = True):
    assert len(lis) == 2
    assert type(name) is str
    df_symm = lis[1].copy()
    df_symm.y = 0.5*(df_symm.y + (lis[0].y.to_numpy()))
    df_symm = pd.DataFrame(list(zip(df_symm.x, df_symm.y)), columns = ['x','y'])
    if show is True:
        dfs = {"Symmetrized "+str(temperature_thickness): df_symm}
        fig = go.Figure()
        fig.update_layout(
        title="Symmetrized longitudinal resistivity",
        xaxis_title= "\u03BC\u2080H (T)",
        yaxis_title="\u03C1\u02E3\u02E3 (\u03BC\u03A9 cm)",
        legend_title="Signal",
        font=dict(
            family="Times New Roman, monospace",
            size=18,
            color="black"))
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
        for i in dfs:
            fig = fig.add_trace(go.Scatter(x = dfs[i].x, y = dfs[i].y, name = i, mode = 'markers'))
        fig.show()
    if save is True:
        df_symm.to_csv(name + '_symmetrized.csv', sep='\t', encoding='utf-8')
    return df_symm

def asymmetrize_list_noflip(lis, temperature_thickness, name, show = True, save = True):
    assert len(lis) == 2
    assert type(name) is str
    df_asymm = lis[1].copy()
    df_asymm.y = 0.5*(df_asymm.y - (lis[0].y.to_numpy()))
    if show is True:
        dfs = {"Anti-symmetrized "+str(temperature_thickness): df_asymm}
        fig = go.Figure()
        fig.update_layout(
        title="Anti-symmetrized transverse resistivity",
        xaxis_title= "\u03BC\u2080H (T)",
        yaxis_title="\u03C1\u02B8\u02E3 (\u03BC\u03A9 cm)",
        legend_title="Signal",
        font=dict(
            family="Times New Roman, monospace",
            size=18,
            color="black"))
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
        for i in dfs:
            fig = fig.add_trace(go.Scatter(x = dfs[i].x, y = dfs[i].y, name = i, mode = 'markers'))
        fig.show()
    if save is True:
        df_asymm.to_csv(name + '_anti_symmetrized.csv', sep='\t', encoding='utf-8')
    return df_asymm

def symmetrize_list_filtered(lis, w_size, polyorder, temperature_thickness, name, show = True, save = True):
    assert len(lis) == 2
    assert type(name) is str
    df_symm = lis[1].copy()
    df_symm.y = 0.5*(df_symm.y + np.flip(lis[0].y.to_numpy()))
    df_symm = pd.DataFrame(list(zip(df_symm.x, df_symm.y)), columns = ['x','y'])
    filtered = signal.savgol_filter(df_symm.y, w_size, polyorder, mode="mirror")
    df_symm_filtered = pd.DataFrame(list(zip(df_symm.x, filtered)), columns = ['x','y'])
    print(df_symm_filtered)
    if show is True:
        dfs = {"Symmetrized "+str(temperature_thickness): df_symm, "SG filtered symmetrized "+str(temperature_thickness): df_symm_filtered}
        fig = go.Figure()
        fig.update_layout(
        title="Symmetrized longitudinal resistivity",
        xaxis_title= "\u03BC\u2080H (T)",
        yaxis_title="\u03C1\u02E3\u02E3 (\u03BC\u03A9 cm)",
        legend_title="Signal",
        font=dict(
            family="Times New Roman, monospace",
            size=18,
            color="black"))
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
        for i in dfs:
            fig = fig.add_trace(go.Scatter(x = dfs[i].x, y = dfs[i].y, name = i, mode = 'markers'))
        fig.show()
    if save is True:
        df_symm.to_csv(name + '_symmetrized.csv', sep='\t', encoding='utf-8')
        df_symm_filtered.to_csv(name + '_.csv', sep='\t', encoding='utf-8')
    return df_symm_filtered

def asymmetrize_list_filtered(lis, w_size, polyorder, temperature_thickness, name, show = True, save = True):
    assert len(lis) == 2
    assert type(name) is str
    df_asymm = lis[1].copy()
    df_asymm.y = 0.5*(df_asymm.y - np.flip(lis[0].y.to_numpy()))
    df_asymm = pd.DataFrame(list(zip(df_asymm.x, df_asymm.y)), columns = ['x','y'])
    filtered = signal.savgol_filter(df_asymm.y, w_size, polyorder, mode="mirror")
    df_asymm_filtered = pd.DataFrame(list(zip(df_asymm.x, filtered)), columns = ['x','y'])
    print(df_asymm_filtered)
    if show is True:
        dfs = {"Anti-symmetrized "+str(temperature_thickness): df_asymm, "SG filtered anti-symmetrized "+str(temperature_thickness): df_asymm_filtered}
        fig = go.Figure()
        fig.update_layout(
        title="Anti-symmetrized transverse resistivity",
        xaxis_title= "\u03BC\u2080H (T)",
        yaxis_title="\u03C1\u02B8\u02E3 (\u03BC\u03A9 cm)",
        legend_title="Signal",
        font=dict(
            family="Times New Roman, monospace",
            size=18,
            color="black"))
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
        for i in dfs:
            fig = fig.add_trace(go.Scatter(x = dfs[i].x, y = dfs[i].y, name = i, mode = 'markers'))
        fig.show()
    if save is True:
        df_asymm.to_csv(name + '_anti_symmetrized.csv', sep='\t', encoding='utf-8')
        df_asymm_filtered.to_csv(name + '_filtered_anti_symmetrized.csv', sep='\t', encoding='utf-8')
    return df_asymm_filtered

def symmetrize_list_noflip_filtered(lis, w_size, polyorder, temperature_thickness, name, show = True, save = True):
    assert len(lis) == 2
    assert type(name) is str
    df_symm = lis[1].copy()
    df_symm.y = 0.5*(df_symm.y + (lis[0].y))
    df_symm = pd.DataFrame(list(zip(df_symm.x, df_symm.y)), columns = ['x','y'])
    filtered = signal.savgol_filter(df_symm.y, w_size, polyorder, mode="mirror")
    df_symm_filtered = pd.DataFrame(list(zip(df_symm.x, filtered)), columns = ['x','y'])
    print(df_symm_filtered)
    if show is True:
        dfs = {"Symmetrized "+str(temperature_thickness): df_symm, "SG filtered symmetrized "+str(temperature_thickness): df_symm_filtered}
        fig = go.Figure()
        fig.update_layout(
        title="Symmetrized longitudinal resistivity",
        xaxis_title= "\u03BC\u2080H (T)",
        yaxis_title="\u03C1\u02E3\u02E3 (\u03BC\u03A9 cm)",
        legend_title="Signal",
        font=dict(
            family="Times New Roman, monospace",
            size=18,
            color="black"))
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
        for i in dfs:
            fig = fig.add_trace(go.Scatter(x = dfs[i].x, y = dfs[i].y, name = i, mode = 'markers'))
        fig.show()
    if save is True:
        df_symm.to_csv(name + '_symmetrized.csv', sep='\t', encoding='utf-8')
        df_symm_filtered.to_csv(name + '_filtered_symmetrized.csv', sep='\t', encoding='utf-8')
    return df_symm_filtered

def asymmetrize_list_noflip_filtered(lis, w_size, polyorder, temperature_thickness, name, show = True, save = True):
    assert len(lis) == 2
    assert type(name) is str
    df_asymm = lis[1].copy()
    df_asymm.y = 0.5*(df_asymm.y - (lis[0].y))
    df_asymm = pd.DataFrame(list(zip(df_asymm.x, df_asymm.y)), columns = ['x','y'])
    filtered = signal.savgol_filter(df_asymm.y, w_size, polyorder, mode="mirror")
    df_asymm_filtered = pd.DataFrame(list(zip(df_asymm.x, filtered)), columns = ['x','y'])
    print(df_asymm_filtered)
    if show is True:
        dfs = {"Anti-symmetrized "+str(temperature_thickness): df_asymm, "SG filtered anti-symmetrized "+str(temperature_thickness): df_asymm_filtered}
        fig = go.Figure()
        fig.update_layout(
        title="Anti-symmetrized transverse resistivity",
        xaxis_title= "\u03BC\u2080H (T)",
        yaxis_title="\u03C1\u02B8\u02E3 (\u03BC\u03A9 cm)",
        legend_title="Signal",
        font=dict(
            family="Times New Roman, monospace",
            size=18,
            color="black"))
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
        for i in dfs:
            fig = fig.add_trace(go.Scatter(x = dfs[i].x, y = dfs[i].y, name = i, mode = 'markers'))
        fig.show()
    if save is True:
        df_asymm.to_csv(name + '_anti_symmetrized.csv', sep='\t', encoding='utf-8')
        df_asymm_filtered.to_csv(name + '_filtered_anti_symmetrized.csv', sep='\t', encoding='utf-8')
    return df_asymm_filtered

def unified_asymm_curves(df):
    fig = go.Figure()
    fig.update_layout(
    title="Anti-symmetrized transverse resistivity",
    xaxis_title= "\u03BC\u2080H (T)",
    yaxis_title="\u03C1\u02B8\u02E3 (\u03BC\u03A9 cm)",
    legend_title="Signal",
    font=dict(
        family="Times New Roman, monospace",
        size=18,
        color="black"))
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
    for i in df:
        fig = fig.add_trace(go.Scatter(x = df[i].x, y = df[i].y, name = i, mode = 'markers'))
    fig.show()


def unified_symm_curves(df):
    fig = go.Figure()
    fig.update_layout(
    title="Symmetrized longitudinal resistivity",
    xaxis_title= "\u03BC\u2080H (T)",
    yaxis_title="\u03C1\u02E3\u02E3 (\u03BC\u03A9 cm)",
    legend_title="Signal",
    font=dict(
        family="Times New Roman, monospace",
        size=18,
        color="black"))
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
    for i in df:
        fig = fig.add_trace(go.Scatter(x = df[i].x, y = df[i].y, name = i, mode = 'markers'))
    fig.show()


def hall_coeff(df, name, save = False):
    df.y = df.y / df.x
    fig = go.Figure()
    fig.update_layout(
    title="Hall coefficient",
    xaxis_title= "\u03BC\u2080H (T)",
    yaxis_title="\u03C1\u02B8\u02E3 (\u03BC\u03A9 cm) / \u03BC\u2080H (T)",
    legend_title="Signal",
    font=dict(
        family="Times New Roman, monospace",
        size=18,
        color="black"))
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
    fig = fig.add_trace(go.Scatter(x = df.x, y = df.y, mode = 'markers'))
    fig.show()
    if save is True:
        df.to_csv("Hall coefficient " + name + ' .csv', sep='\t', encoding='utf-8')