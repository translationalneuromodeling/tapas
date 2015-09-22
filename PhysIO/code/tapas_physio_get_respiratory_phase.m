function [rphase, fh] = tapas_physio_get_respiratory_phase(pulset, rsampint,...
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
%
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
%
% $Id: tapas_physio_get_respiratory_phase.m 534 2014-08-28 18:05:58Z kasperla $

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
kernel = [ones(1, ksize)*-1 0 ones(1, ksize)];
dpulse = -conv(pulset, kernel);
dpulse = dpulse(ksize+1:end-ksize);

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
    fh = plot_traces(pulset, rsampint, rout, resp_max, cumsumh, sumh, h, ...
        npulse, dpulse, rphase);
else
    fh = [];
end

end

function fh = plot_traces(pulset, rsampint, rout, resp_max, ...
    cumsumh, sumh, h, npulse, dpulse, rphase)

nsamples = length(pulset);
t = (0:nsamples-1)*rsampint;
feqht = cumsumh/sumh*pi;

fh = tapas_physio_get_default_fig_params();
set(fh, 'Name', ...
   'get_respiratory_phase: histogram for respiratory phase estimation');

hs(1) = subplot(2,2,1);
plot(t,pulset); 
xlabel('t (s)'); 
ylabel('breathing amplitude (a. u.)'); 
title('(filtered) breathing time series');

if resp_max < inf 
    hold on; 
    plot(t, ones(size(t)) * resp_max, 'k--');
    hold on; 
    hp = plot(t, -ones(size(t)) * resp_max, 'k--');        
    legend(hp, ...
        'threshold for maximum amplitude to be considered in histogram');    
    set(gcf, 'Name', ...
        [get(gcf, 'Name') ' - with amplitude overshoot-correction']);
end

hs(2) = subplot(2,2,2);
bar(rout, h); 
xlabel('normalized breathing amplitude'); 
ylabel('counts');
title('histogram for phase mapping');
xlim([-0.1 1.1]);

hs(3) = subplot(2,2,3); plot(rout, [feqht, cos(feqht), sin(feqht)]); 
xlabel('normalized breathing amplitude'); 
title(...
    'equalized histogram bin amplitude -> phase transfer function (f_{eqht})');
legend('f: normalized amplitude -> phase transfer function', 'cos(f)', ...
    'sin(f)', 'Location', 'NorthWest');

%figure('Name', 'Histogram: Respiration phase estimation');
hs(4) = subplot(2,2,4);
plot(t, [npulse*10, dpulse, (rphase-pi)]);
legend('10*normalized breathing belt amplitude', ...
    '-1 = exhale, 1 = inhale', 'estimated respiratory phase');    
ylim([-0.2 10.2]);
title('Histogram-based respiration phase estimation');

linkaxes(hs([1 4]), 'x');
linkaxes(hs([2 3]), 'x');

end
