function [rphase, pulset, rsampint, rout, resp_max, cumsumh, sumh, h, ...
        npulse, dpulse, fh] = tapas_physio_get_respiratory_phase(pulset, rsampint,...
    verbose, thresh)

% get_respiratory_phase is a function for creating respiratory phase regressor.
% from physiological monitoring files acquired using spike, that
% can be included in SPM5 design matrices.
%
% FORMAT [rphase] = tapas_physio_get_respiratory_phase(pulset,rsampint)
%
% Inputs: 
%        pulset - respiratory belt data read from spike file
%        rsampint - sampling frequency
%        verbose - 1 for graphical output (debugging mainly)
%        thresh.resp_max -  if set, all peaks above that breathing belt amplitude 
%                        are ignored for respiratory phase histogram evaluation
%
% Outputs:
%        rphase - respiratory phase when each slice of each volume was
%        acquired
%
% The regressors are calculated in the following way as described in
% Glover et al, 2000, MRM, (44) 162-167: 
% 1) Find the max and min amplitudes from belt response
% 2) Normalise the respiratory amplitude
% 3) Calculate the histogram from the number of occurences of specific respiratory
%    amplitudes in bins 1:100.
% 4) Calculate the running integral of the histogram for the bin
%    corresponding to each respiratory amplitude
%_______________________________________________________________________

% Author: Lars Kasper
%         Error handling for temporary breathing belt failures:
%         Eduardo Aponte, TNU Zurich
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


%% get histogram of amplitudes of breathing belt signal
if nargin < 3
    verbose = false;
end

if nargin < 4
    resp_max = inf;
elseif ~isfield(thresh, 'resp_max')
    resp_max = inf;
elseif isempty(thresh.resp_max)
    resp_max = inf;
else 
    resp_max = thres.resp_max;   
end

% Check input

assert(all(abs(pulset) ~= inf), ...
    'Infinity values in the respiratory regressor');
assert(~any(isnan(pulset)), 'Nan values in the respiratory regressors');

% Weird line...
overshoot = find(abs(pulset) > resp_max);
pulset(overshoot) = resp_max;

maxr = max(pulset);
minr = min(pulset);

% Compute normalized signal and the sign of the derivative

npulse = (pulset-minr)/(maxr-minr);

% Calculate derivative of normalised pulse wrt time
% over 1 sec of data as described in Glover et al.
ksize = round(0.5 * (1/rsampint));
kernel = ksize:-1:-ksize; kernel = kernel ./ sum(kernel.^2);
dpulse = tapas_physio_conv(pulset, kernel, 'symmetric');
% This uses a quadratic Savitzky-Golay filter, for which the coefficients
% have a simple linear form. See e.g.
% https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
% `scipy.signal.savgol_coeffs(5, polyorder=2, deriv=1, use='conv')`

% Tolerance to the derivative
dpulse(abs(dpulse) < 1e-4) = 0;
dpulse = sign(dpulse);

% number of histogram bins determined by dynamic range of detected values
% and length of input time course
nbins = min(length(unique(pulset)), floor(length(pulset)/100));

[h, rout] = hist(npulse(dpulse ~=0 & npulse < resp_max), nbins);

binnum = floor(npulse*(nbins-1)) + 1;
binnum(overshoot) = nbins;

cumsumh = cumsum(h');
sumh = cumsumh(end);

dpulse(dpulse == 0) = nan;
rphase = pi*(cumsumh(binnum)/sumh).*dpulse+pi;

if verbose
    fh = tapas_physio_plot_traces(pulset, rsampint, rout, resp_max, cumsumh, sumh, h, ...
        npulse, dpulse, rphase);
else
    fh = [];
end

end


