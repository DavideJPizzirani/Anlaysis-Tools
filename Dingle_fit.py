import numpy as np
import pandas as pd
from lmfit import Model
from lmfit import conf_interval, conf_interval2d, report_ci
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import scipy.optimize as opt
import statistics
import matplotlib.pylab as plt
import copy
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Minimizer, create_params, fit_report
import plotly.graph_objects as go
from plotly.offline import iplot
import plotly.express as px
import seaborn as sns
from typing import Optional
import scipy.constants

def convert_pd_dataframe_into_np_arrays_x(df):
    x_array = df.x.to_numpy()
    return x_array

def convert_pd_dataframe_into_np_arrays_y(df):
    y_array = df.y.to_numpy()
    return y_array

xi = 14.694  # T/K


def plot_results_brute(result, best_vals=True, varlabels=None,
                       output=None):
    """Visualize the result of the brute force grid search.

    The output file will display the chi-square value per parameter and contour
    plots for all combination of two parameters.

    Inspired by the `corner` package (https://github.com/dfm/corner.py).

    Parameters
    ----------
    result : :class:`~lmfit.minimizer.MinimizerResult`
        Contains the results from the :meth:`brute` method.

    best_vals : bool, optional
        Whether to show the best values from the grid search (default is True).

    varlabels : list, optional
        If None (default), use `result.var_names` as axis labels, otherwise
        use the names specified in `varlabels`.

    output : str, optional
        Name of the output PDF file (default is 'None')
    """
    npars = len(result.var_names)
    _fig, axes = plt.subplots(npars, npars)

    if not varlabels:
        varlabels = result.var_names
    if best_vals and isinstance(best_vals, bool):
        best_vals = result.params

    for i, par1 in enumerate(result.var_names):
        for j, par2 in enumerate(result.var_names):

            # parameter vs chi2 in case of only one parameter
            if npars == 1:
                axes.plot(result.brute_grid, result.brute_Jout, 'o', ms=3)
                axes.set_ylabel(r'$\chi^{2}$')
                axes.set_xlabel(varlabels[i])
                if best_vals:
                    axes.axvline(best_vals[par1].value, ls='dashed', color='r')

            # parameter vs chi2 profile on top
            elif i == j and j < npars-1:
                if i == 0:
                    axes[0, 0].axis('off')
                ax = axes[i, j+1]
                red_axis = tuple(a for a in range(npars) if a != i)
                ax.plot(np.unique(result.brute_grid[i]),
                        np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        'o', ms=3)
                ax.set_ylabel(r'$\chi^{2}$')
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_ticks_position('right')
                ax.set_xticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='r')

            # parameter vs chi2 profile on the left
            elif j == 0 and i > 0:
                ax = axes[i, j]
                red_axis = tuple(a for a in range(npars) if a != i)
                ax.plot(np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        np.unique(result.brute_grid[i]), 'o', ms=3)
                ax.invert_xaxis()
                ax.set_ylabel(varlabels[i])
                if i != npars-1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel(r'$\chi^{2}$')
                if best_vals:
                    ax.axhline(best_vals[par1].value, ls='dashed', color='r')

            # contour plots for all combinations of two parameters
            elif j > i:
                ax = axes[j, i+1]
                red_axis = tuple(a for a in range(npars) if a not in (i, j))
                X, Y = np.meshgrid(np.unique(result.brute_grid[i]),
                                   np.unique(result.brute_grid[j]))
                lvls1 = np.linspace(result.brute_Jout.min(),
                                    np.median(result.brute_Jout)/2.0, 7, dtype='int')
                lvls2 = np.linspace(np.median(result.brute_Jout)/2.0,
                                    np.median(result.brute_Jout), 3, dtype='int')
                lvls = np.unique(np.concatenate((lvls1, lvls2)))
                ax.contourf(X.T, Y.T, np.minimum.reduce(result.brute_Jout, axis=red_axis),
                            lvls, norm=LogNorm())
                ax.set_yticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='r')
                    ax.axhline(best_vals[par2].value, ls='dashed', color='r')
                    ax.plot(best_vals[par1].value, best_vals[par2].value, 'rs', ms=3)
                if j != npars-1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel(varlabels[i])
                if j - i >= 2:
                    axes[i, j].axis('off')

    if output is not None:
        plt.savefig(output)


def prepare_dataset_to_dingle_fit(df, order, m, T):
    print("Remember: x must be 1/B!")
    assert type(order) is int and order in [0, 1, 2]
    if order == 0:
        x = convert_pd_dataframe_into_np_arrays_x(df)
        ampl = convert_pd_dataframe_into_np_arrays_y(df)
        y = ampl * (1/x ** (-1/2)) / ((xi * m * T * x) / (np.sinh(xi * m * T * x)))
        log_y = np.log(y)
        new = np.vstack((x, log_y)).T
        new_df = pd.DataFrame(new, columns=['x', 'y'])
        dfs = {"Dingle dataset": new_df}
        fig = go.Figure()
        fig.update_layout(
        title="Dingle plot",
        xaxis_title= " 1/\u03BC\u2080H (T)",
        yaxis_title="ln(Amplitude H^(1/2)/Rt)",
        legend_title="Dingle plot on raw data",
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
    if order == 1:
        x = convert_pd_dataframe_into_np_arrays_x(df)
        ampl = convert_pd_dataframe_into_np_arrays_y(df)
        y = ampl * (1/x ** (-1/2)) / ((xi * m * T * x) / (np.sinh(xi * m * T * x)))
        log_y = np.log(y)
        new = np.vstack((x, log_y)).T
        new_df = pd.DataFrame(new, columns=['x', 'y'])
        dfs = {"Dingle dataset": new_df}
        fig = go.Figure()
        fig.update_layout(
        title="Dingle plot",
        xaxis_title= " 1/\u03BC\u2080H (T)",
        yaxis_title="ln(Amplitude H^(1/2)/Rt)",
        legend_title="Dingle plot on Ist derived signal",
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
    if order == 2:
        x = convert_pd_dataframe_into_np_arrays_x(df)
        ampl = convert_pd_dataframe_into_np_arrays_y(df)
        y = ampl * (1/x ** (-1/2)) / ((xi * m * T * x) / (np.sinh(xi * m * T * x)))
        log_y = np.log(y)
        new = np.vstack((x, log_y)).T
        new_df = pd.DataFrame(new, columns=['x', 'y'])
        dfs = {"Dingle dataset": new_df}
        fig = go.Figure()
        fig.update_layout(
        title="Dingle plot",
        xaxis_title= " 1/\u03BC\u2080H (T)",
        yaxis_title="ln(Amplitude H^(1/2)/Rt)",
        legend_title="Dingle plot on Ist derived signal",
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
    return new_df

def fit_dingle(df, m, value, value_Td, min_Td, max_Td, value_a, min_a, max_a):
    # Data
    x_values = df.x
    y_values = df.y 
    print(np.std(y_values))
    # Model Dingle
    def model_dingle_brute(params, x_values, y_values):
        xi = 14.693  # T/K
        a = params['a']
        Td = params['Td']
        intercept = a
        model = (-xi * m * Td) * x_values + intercept
        return (model - y_values)

    #Specify parameters and print them
    params = create_params(a=dict(value=value_a, vary=True, min=min_a, max=max_a),
                        Td=dict(value=value_Td, vary=True, min=min_Td, max=max_Td))

    params.pretty_print()

    # Initialize and perform the grid search
    fitter = Minimizer(model_dingle_brute, params, fcn_args=(x_values, y_values), nan_policy='omit')
    result_brute = fitter.minimize(method='brute', Ns=25, keep=25)
    print(fit_report(result_brute))
    best_result = copy.deepcopy(result_brute)
    for candidate in result_brute.candidates:
        trial = fitter.minimize(method='leastsq', params=candidate.params)
        if trial.chisqr < best_result.chisqr:
            best_result = trial
    print(fit_report(best_result))

    #interacting plot
    df = np.vstack((x_values, y_values)).T
    a = model_dingle_brute(result_brute.params, x_values, y_values)
    data_and_model=y_values+a
    best_brute_fit = np.vstack((x_values, data_and_model)).T
    b = model_dingle_brute(best_result.params, x_values, y_values)
    c=y_values+b
    best_lstq_fit = np.vstack((x_values, c)).T
    df_original_data = pd.DataFrame(df, columns=['x','y'])
    df_best_brute_fit = pd.DataFrame(best_brute_fit, columns=['x','y'])
    df_best_lstq_fit = pd.DataFrame(best_lstq_fit, columns=['x','y'])
    fig = go.Figure()
    fig.update_layout(
    title="Dingle plot and fit",
    xaxis_title= "\u03BC\u2080H (1/T)",
    yaxis_title="\u0394\u03C1\u02E3\u02E3 (\u03BC\u03A9 cm)",
    legend_title="Legend",
    font=dict(
        family="Times New Roman, monospace",
        size=18,
        color="black"))
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
    fig = fig.add_trace(go.Scatter(x = df_original_data.x, y = df_original_data.y, name = 'Data', mode = 'markers', error_y=dict(
            type='percent', # value of error bar given as percentage of y value
            value=value,
            visible=True)))
    fig = fig.add_trace(go.Scatter(x = df_best_brute_fit.x, y = df_best_brute_fit.y, name = 'Brute fit', mode = 'lines'))
    fig = fig.add_trace(go.Scatter(x = df_best_lstq_fit.x, y = df_best_lstq_fit.y, name = 'Lstq', mode = 'lines'))
    fig.show()
    print(best_result.params.valuesdict())
    T_d = best_result.params.get('Td')
    t_q = scipy.constants.hbar/(2*np.pi*scipy.constants.k*T_d)
    return t_q

    
    


    #plt.plot(x_values, y_values, 'o')
    #plt.plot(x_values, y_values + model_dingle_brute(result_brute.params, x_values, y_values), '-',
    #     label='brute')
    #plt.plot(x_values, y_values + model_dingle_brute(best_result.params, x_values, y_values), '--',
    #     label='brute followed by leastsq')
    #plt.legend()
    #plt.show()
    