#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2019


''''

Plotting tools for distribution analysis.

'''

import numpy as np
from scipy.integrate import cumtrapz

COMP = 0
INCOMP = 1


def compute_predicted_error_quantiles(fits, quantiles=None):
    '''Compute the quantiles. '''

    if quantiles is None:
        quantiles = [0.25, 0.50, 0.75, 1.0]

    # Get the unique trial types, which are coded as the second index
    # The first index is always the action
    trial_types = set([field[1] for field in fits.__class__.fields])

    # Compute the cumulative distribution
    delta_er = {'quantiles': quantiles}

    for trial in trial_types:
        # Compute the cumulative distribution
        cums = cumtrapz(fits.__dict__[COMP, trial] +
                fits.__dict__[INCOMP, trial], fits.time)
        total = cums[-1]
        cums /= total

        cums_comp = cumtrapz(fits.__dict__[COMP, trial], fits.time)
        cums_comp /= total

        delta_er[trial] = {'x': [], 'y': []}
        i0 = 0
        for quantile in quantiles:
            # Find the index for this quantile
            i = np.searchsorted(cums, quantile) - 1

            # Compute the total mass for this quantile
            total_mass = (cums[i] - cums[i0])

            # Compute the fraction corresponding to comps
            comp_mass = (cums_comp[i] - cums_comp[i0])

            x = (fits.time[i] + fits.time[i0])/2.0
            delta_er[trial]['x'].append(x)
            delta_er[trial]['y'].append(comp_mass/total_mass)

            i0 = i

    return delta_er


def compute_mean_rt_by_quantiles(fit, time, quantiles):
    '''Compute the mean rt for each quantile. '''

    # Cumulative density function
    cum = cumtrapz(fit, time)
    # Normalize if necessary
    fit /= cum[-1]
    cum /= cum[-1]

    i0 = 0

    rts = []

    for quantile in quantiles:
        # Find the index for this quantile
        i = np.searchsorted(cum, quantile) - 1

        # Normalization
        mass = cum[i] - cum[i0]

        # Mean reaction time
        rt = np.trapz(fit[i0:i] * time[i0:i], time[i0:i]) / mass

        rts.append(rt)

        # Update the index
        i0 = i

    return rts


def compute_predicted_rt_quantiles(fits, comp, incomp, quantiles=None):
    '''Compute the rt and er the respective quantiles. '''

    if quantiles is None:
        quantiles = [0.25, 0.50, 0.75, 1.0]

    quantiles = sorted(quantiles)

    # Compute the cumulative distribution
    delta_rt = {'quantiles': quantiles, 'x': [], 'y': []}

    fit_comp = fits.__dict__[COMP, comp[1]] #+ \
            #fits.__dict__[INCOMP, comp[1]]
    fit_incomp = fits.__dict__[INCOMP, incomp[1]] #+ \
            #fits.__dict__[COMP, incomp[1]]

    time = fits.time

    rts_comp = compute_mean_rt_by_quantiles(fit_comp, time, quantiles)
    rts_incomp = compute_mean_rt_by_quantiles(fit_incomp, time, quantiles)

    rts_comp = np.array(rts_comp)
    rts_incomp = np.array(rts_incomp)

    delta_rt['x'] = (rts_comp + rts_incomp)/2.0
    delta_rt['y'] = rts_incomp - rts_comp

    return delta_rt

# Empirical plots


def compute_empirical_delta_rt(rt, acts, tt, quants=None):

    if quants is None:
        quants = [0.0, 0.2, 0.4, 0.6, 0.8, 0.99]

    pers = np.array(quants) * 1e2
    comp_rt = rt[np.logical_and(acts == tt, acts == COMP)]
    # comp_rt = rt[tt == COMP]
    comp_pers = np.percentile(comp_rt, pers)

    incomp_rt = rt[np.logical_and(acts == tt, acts == INCOMP)]
    # incomp_rt = rt[tt == INCOMP]
    incomp_pers = np.percentile(incomp_rt, pers)

    x = []
    y = []

    for i in range(1, len(pers)):
        comp_t = comp_rt[np.logical_and(comp_pers[i-1] < comp_rt,
                comp_rt < comp_pers[i])]
        incomp_t = incomp_rt[np.logical_and(incomp_pers[i-1] < incomp_rt,
                incomp_rt < incomp_pers[i])]

        #  x.append(np.mean(np.concatenate([comp_t, incomp_t])))
        comp_t = np.mean(comp_t)
        incomp_t = np.mean(incomp_t)
        x.append((comp_t + incomp_t)/2)
        y.append(incomp_t - comp_t)

    return x, y


def plot_empirical_delta_rt(ax, rt, acts, tt, sign=1.0, pers=None,
        pdict=None):
    '''Make delta plots from the reaction time.

    :param ax -- Axis.
    :param rt      -- Array of reaction times.
    :acts acts      -- Array of actions
    :param tt       -- Array of trial types
    :param sign     -- Sign of the plot
    :param pers     -- Percentiles. Defaults to [0, 20, 40, 60, 80, 99.9]
    :param pdict    -- Dictionary of arguments for plotting. Defaults to {}

    '''

    if pdict is None:
        pdict = {}

    if pers is None:
        pers = [0, 20, 40, 60, 80, 99.9]

    # Correct trials

    x, y = compute_empirical_delta_rt(rt, acts, tt, np.array(pers)/1e2)

    ax.plot(np.array(x), sign * np.array(y), **pdict)

    return x, y


def compute_empirical_delta_er(rt, acts, tt, pers):

    pers = np.percentile(rt, pers)

    t0 = 0
    er, tbins = [], []

    for perct in pers:
        # Recover index of the elements
        index = np.logical_and(t0 <= rt, rt < perct)

        # Get the denominator, if empty, write a none
        n = np.sum(index)
        if n == 0:
            val = None
        else:
            # Error rate
            val = np.sum(acts[index] == tt[index])/float(n)

        er.append(val)
        #tbins.append(np.mean(rt[np.logical_and(index, acts == tt)]))
        tbins.append(np.mean(rt[index]))

        t0 = perct

    return tbins, er


def plot_empirical_delta_er(ax, rt, acts, tt, scale=None, pers=None, 
        pdict=None):
    '''

    correct     -- Correct action
    rt          -- All reaction times
    acts        -- All actions
    tt          -- Trial types
    sclale      -- A function to apply to the values. If None its the 
                    identity function.

    '''

    if pdict is None:
        pdict = {}

    if pers is None:
        pers = [20, 40, 60, 80, 99.9]

    if scale is None:
        scale = lambda x: x

    tbins, er = compute_empirical_delta_er(rt, acts, tt, pers)
    
    ax.plot(tbins, scale(er), **pdict)

    return


if __name__ == '__main__':
    pass
