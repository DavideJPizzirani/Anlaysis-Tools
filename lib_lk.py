import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial import Polynomial
from functional import seq


xi = 14.693  # T/K


def nyquist_freq_for_qo(B0, tesla_per_min, dt=0.3):
    return B0**2 / (2*tesla_per_min/60*dt)


def bstar_naive(B0, B1):
    # use inverse average of B0 and B1
    # where B0 is an indication of where the Q.O. start
    return 1 / ((1/B0 + 1/B1) / 2)


def fit_mass_naive(T, ampl, Bstar, sigma=None):
    def model(T, C, m):
        if m == 0:
            return C
        else:
            x = 1 / Bstar
            return C * xi * m * T * x / np.sinh(xi * m * T * x)

    pfit, pcov = curve_fit(
        model,
        xdata=T,
        ydata=ampl,
        sigma=sigma,
        p0=[1, 0.1],
        bounds=(0, np.inf)
    )
    perr = np.sqrt(np.diag(pcov))

    return pfit, perr, model


def fit_mass_naive_normalised(T, ampl, Bstar, sigma=None):
    def model(T, m):
        if m == 0:
            return 1
        else:
            x = 1 / Bstar
            return xi * m * T * x / np.sinh(xi * m * T * x)

    pfit, pcov = curve_fit(
        model,
        xdata=T,
        ydata=ampl,
        sigma=sigma,
        p0=[0.1],
        bounds=(0, np.inf)
    )
    perr = np.sqrt(np.diag(pcov))

    return pfit, perr, model


def fit_mass_integral(T, T0, ampl, Td, B0, B1, dv=1):
    def integrand(x, T, m):
        if x == 0 or m == 0 or T == 0:
            R_T = 1
        else:
            R_T = xi * m * T * x / np.sinh(xi * m * T * x)
        R_D = np.exp(-xi * m * Td * x)
        window = 0.5 * (1 - np.cos(2 * np.pi / (1/B0 - 1/B1)) * (x - 1/B0))
        return x ** (2*dv - 1/2) * R_T * R_D * window

    def scalar_model(T, C, m):
        integral, _ = quad(integrand, a=1/B1, b=1/B0, args=(T, m))
        normalisation, _ = quad(integrand, a=1/B1, b=1/B0, args=(T0, m))
        return C * integral / normalisation

    model = np.vectorize(scalar_model)

    pfit, pcov = curve_fit(
        model,
        xdata=T,
        ydata=ampl,
        p0=[1, 0.1],
        bounds=([0, 0], [10, 1])
    )
    perr = np.sqrt(np.diag(pcov))

    return pfit, perr, model


def remove_bg(df, order=2):
    df1 = df.copy()
    p = fit_bg(df1, order)
    df1.y -= p(df1.x)
    return df1


def fit_bg(df, order=2):
    coef = polyfit(df.x, df.y, deg=order)
    p = Polynomial(coef)
    return p


def remove_bg_invert_x(df, xstart, xend, order=2):
    df = df[df.x.between(xstart, xend)]

    poly = fit_bg(df, order=order)

    df_no_bg = df.copy()
    df_no_bg.y -= poly(df_no_bg.x)
    df_no_bg.x = 1 / df_no_bg.x

    return df_no_bg


def dataset_remove_bg(dataset, xstart, xend, order=2):
    new_dataset = []
    for df in dataset:
        df1 = df.copy()
        new_dataset.append(remove_bg(df1[df1.x.between(xstart, xend)], order))
    return new_dataset


def dataset_remove_bg_invert_x(dataset, xstart, xend, order=2):
    new_dataset = []
    for df in dataset:
        df1 = df.copy()
        new_dataset.append(remove_bg_invert_x(df1, xstart, xend, order))
    return new_dataset


def prepare_dingle_fit(x, ampl, m, T, dv=0):
    # Full LK prefactor expression:
    # y = A x^p 1/sinh(xi m T x) exp(-xi m Td x)
    # where p depends on the derivative order: p=2 dv - 1/2
    x = np.array(x)

    # We have to divide out x^p 1/sinh(xi m T x)
    y = np.array(ampl) / (x ** (2*dv - 1/2) / np.sinh(xi * m * T * x))

    # Take the log to obtain -xi m T x + Log A (some offset)
    y = np.log(ampl)

    return y


def fit_dingle(x, log_ampl, m):
    def model(x, Td, A):
        return -xi * m * Td * x + A

    pfit, pcov = curve_fit(
        model,
        xdata=x,
        ydata=log_ampl,
        p0=[5, 0],
        bounds=(0, 50)
    )
    perr = np.sqrt(np.diag(pcov))

    return pfit, perr, model


def model_lk(x, T, ampl, freq, m, Td, phase):
    R_T = xi * m * T * x / np.sinh(xi * m * T * x)
    R_D = np.exp(-xi * m * Td * x)
    return ampl * np.sqrt(x) * R_T * R_D * np.cos(2 * np.pi * freq * x + phase)


def fit_lk_phase_1(df, T, freq, m, Td):
    def model(x, ampl, phase):
        return model_lk(x, T, ampl, freq, m, Td, phase)

    pfit, pcov = curve_fit(
        model,
        xdata=df.x,
        ydata=df.y,
        p0=[0.1, 0],
        bounds=([-np.inf, 0], [np.inf, 2 * np.pi])
    )
    perr = np.sqrt(np.diag(pcov))

    return pfit, perr, model


'''def fit_lk_1(dataset, temperatures, freq):
    # this fits all temperature curves at the same time using a single mass

    def model(x_obj, ampl, phase, m, Td):
        y = []
        for x, temp in zip(x_obj.x, temperatures):
            y.append(model_lk(x, temp, ampl, freq, m, Td, phase))
        y = np.concatenate(y)
        return y

    # we want to plot the curves 1 by 1
    def model_for_plot(x, T, ampl, phase, m, Td):
        return model_lk(x, T, ampl, freq, m, Td, phase)

    # concatenate all datasets
    df_y = seq(dataset).map(lambda df: df.y).to_list()
    y = np.concatenate(df_y)

    # x can be an arbitrary object
    x = seq(dataset).map(lambda df: df.x).to_list()

    class AvoidLengthCheck:
        def __init__(self, x) -> None:
            self.x = x

    # fuck you python
    x_obj = AvoidLengthCheck(x)

    pfit, pcov = curve_fit(
        model,
        xdata=x_obj,
        ydata=y,
        p0=[0.1, 0, 0.1, 15],
        bounds=([-np.inf, 0, 0.001, 0], [np.inf, 2 * np.pi, 10, 100])
    )
    perr = np.sqrt(np.diag(pcov))

    return pfit, perr, model_for_plot


def fit_lk_2(dataset, temperatures, freq1, freq2):
    # this fits all temperature curves at the same time using a single mass

    def model(x_obj, ampl1, ampl2, phase1, phase2, m1, m2, Td1, Td2):
        y = []
        for x, temp in zip(x_obj.x, temperatures):
            y.append(model_lk(x, temp, ampl1, freq1, m1, Td1, phase1) +
                     model_lk(x, temp, ampl2, freq2, m2, Td2, phase2))
        y = np.concatenate(y)
        return y

    # we want to plot the curves 1 by 1
    def model_for_plot(x, T, ampl1, ampl2, phase1, phase2, m1, m2, Td1, Td2):
        return model_lk(x, T, ampl1, freq1, m1, Td1, phase1) + model_lk(x, T, ampl2, freq2, m2, Td2, phase2)

    # concatenate all datasets
    df_y = seq(dataset).map(lambda df: df.y).to_list()
    y = np.concatenate(df_y)

    # x can be an arbitrary object
    x = seq(dataset).map(lambda df: df.x).to_list()

    class AvoidLengthCheck:
        def __init__(self, x) -> None:
            self.x = x

    # fuck you python
    x_obj = AvoidLengthCheck(x)

    pfit, pcov = curve_fit(
        model,
        xdata=x_obj,
        ydata=y,
        # assume that the ampl should be O(1) : less than 100 should be okay
        # ampl1 ampl2 phase1 phase2 m1 m2 Td1 Td2
        p0=[0.1, 0.1, 0, 0, 0.1, 0.1, 15, 15],
        bounds=([-100, -100, 0, 0, 0.001, 0.001, 0, 0],
                [100, 100, 2 * np.pi, 2 * np.pi, 10, 10, 100, 100])
    )
    perr = np.sqrt(np.diag(pcov))

    return pfit, perr, model_for_plot

def fit_lk_3(dataset, temperatures, freq1, freq2, freq3):
    # this fits all temperature curves at the same time using a single mass

    def model(x_obj, ampl1, ampl2, ampl3, phase1, phase2, phase3, m1, m2, m3, Td1, Td2, Td3):
        y = []
        for x, temp in zip(x_obj.x, temperatures):
            y.append(model_lk(x, temp, ampl1, freq1, m1, Td1, phase1) +
                     model_lk(x, temp, ampl2, freq2, m2, Td2, phase2) +
                     model_lk(x, temp, ampl3, freq3, m3, Td3, phase3) )
        y = np.concatenate(y)
        return y

    # we want to plot the curves 1 by 1
    def model_for_plot(x, T, ampl1, ampl2, ampl3, phase1, phase2, phase3, m1, m2, m3, Td1, Td2, Td3):
        return model_lk(x, T, ampl1, freq1, m1, Td1, phase1) + model_lk(x, T, ampl2, freq2, m2, Td2, phase2) + model_lk(x, T, ampl3, freq3, m3, Td3, phase3)

    # concatenate all datasets
    df_y = seq(dataset).map(lambda df: df.y).to_list()
    y = np.concatenate(df_y)

    # x can be an arbitrary object
    x = seq(dataset).map(lambda df: df.x).to_list()

    class AvoidLengthCheck:
        def __init__(self, x) -> None:
            self.x = x

    # fuck you python
    x_obj = AvoidLengthCheck(x)

    pfit, pcov = curve_fit(
        model,
        xdata=x_obj,
        ydata=y,
        # assume that the ampl should be O(1) : less than 100 should be okay
        # ampl1 ampl2 phase1 phase2 m1 m2 Td1 Td2
        p0=[0.1, 0.1, 0.1, 0, 0, 0, 0.1, 0.1, 0.1, 15, 15, 15],
        bounds=([-100, -100, -100, 0, 0, 0, 0.001, 0.001, 0.001, 0, 0, 0],
                [100, 100, 100, 2 * np.pi, 2 * np.pi, 2 * np.pi, 10, 10, 10, 100, 100, 100])
    )
    perr = np.sqrt(np.diag(pcov))

    return pfit, perr, model_for_plot'''
