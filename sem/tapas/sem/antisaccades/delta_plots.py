#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2019


''''

Plotting tools for distribution analysis.

'''

import numpy as np
from scipy.integrate import cumtrapz

PRO = 0
ANTI = 1


def compute_error_quantiles(fits, quantiles=None):
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
        cums = cumtrapz(fits.__dict__[PRO, trial] +
                fits.__dict__[ANTI, trial], fits.time)
        total = cums[-1]
        cums /= total

        cums_pro = cumtrapz(fits.__dict__[PRO, trial], fits.time)
        cums_pro /= total

        delta_er[trial] = {'x': [], 'y': []}
        i0 = 0
        for quantile in quantiles:
            # Find the index for this quantile
            i = np.searchsorted(cums, quantile) - 1

            # Compute the total mass for this quantile
            total_mass = (cums[i] - cums[i0])

            # Compute the fraction corresponding to pros
            pro_mass = (cums_pro[i] - cums_pro[i0])

            x = (fits.time[i] + fits.time[i0])/2.0
            delta_er[trial]['x'].append(x)
            delta_er[trial]['y'].append(pro_mass/total_mass)

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


def compute_rt_quantiles(fits, cong, incong, quantiles=None):
    '''Compute the rt and er the respective quantiles. '''

    if quantiles is None:
        quantiles = [0.25, 0.50, 0.75, 1.0]

    quantiles = sorted(quantiles)

    # Compute the cumulative distribution
    delta_rt = {'quantiles': quantiles, 'x': [], 'y': []}

    fit_cong = fits.__dict__[0, cong[1]] + fits.__dict__[1, cong[1]]
    fit_incong = fits.__dict__[0, incong[1]] + fits.__dict__[1, incong[1]]

    time = fits.time

    rts_cong = compute_mean_rt_by_quantiles(fit_cong, time, quantiles)
    rts_incong = compute_mean_rt_by_quantiles(fit_incong, time, quantiles)

    rts_cong = np.array(rts_cong)
    rts_incong = np.array(rts_incong)

    delta_rt['x'] = (rts_cong + rts_incong)/2.0
    delta_rt['y'] = rts_incong - rts_cong

    return delta_rt


if __name__ == '__main__':
    pass
