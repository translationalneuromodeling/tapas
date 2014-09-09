function [rphase, fh] = tapas_physio_get_respiratory_phase(pulset,rsampint,...
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
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_get_respiratory_phase.m 413 2014-01-21 01:26:20Z kasperla $

%% get histogram of amplitudes of breathing belt signal
if nargin < 3
    verbose = false;
end

correct_resp_overshoot = (nargin > 3) && isfield(thresh, 'resp_max') && ~isempty(thresh.resp_max);
    
if correct_resp_overshoot
    pulset_save = pulset;
    iRespOvershoot = find(abs(pulset)>thresh.resp_max);
    pulset(iRespOvershoot) = thresh.resp_max;%[];
end

maxr=max(pulset);
minr=min(pulset);
normpulse=(pulset-minr)/(maxr-minr)*2-1;
nbins=min(length(unique(pulset)), floor(length(pulset)/100)); % 100 % 360
[h, rout] = hist(normpulse, nbins);

if correct_resp_overshoot
%    pulset = pulset_save;
    normpulse=(pulset-minr)/(maxr-minr)*2-1;
end

% Calculate derivative of normalised pulse wrt time
% over 1 sec of data as described in Glover et al.
ksize=round(0.5*(1/rsampint));
kernel=[ones(1,ksize)*-1 0 ones(1,ksize)];
dnormpulse=-conv(normpulse,kernel);
dnormpulse=dnormpulse(ksize+1:end-ksize);
n=find(abs(dnormpulse)==0);
if ~isempty(n)
    dnormpulse(n)=1;
end
dnormpulse=dnormpulse./abs(dnormpulse);

binnum = (floor((normpulse-min(normpulse))/(max(normpulse)-min(normpulse))*(nbins-1)) + 1);

if correct_resp_overshoot
    binnum(iRespOvershoot) = nbins;
end
cumsumh = cumsum(h');
sumh = cumsumh(end);
rphase=pi*(cumsumh(binnum)/sumh).*dnormpulse+pi;


if nargin < 3, verbose = 0; end
if verbose
    fh = tapas_physio_get_default_fig_params();
    set(fh, 'Name', 'get_respiratory_phase: histogram for respiratory phase estimation');
    Nsamples = length(pulset);
    t = (0:Nsamples-1)*rsampint;
    hs(1) = subplot(2,2,1);
    plot(t,pulset); xlabel('t (s)'); ylabel('breathing amplitude (a. u.)'); title('(filtered) breathing time series');
    if correct_resp_overshoot
        hold on; plot(t, ones(size(t))*thresh.resp_max, 'k--');
        hold on; hp = plot(t, -ones(size(t))*thresh.resp_max, 'k--');        
       legend(hp,'threshold for maximum amplitude to be considered in histogram');    
       set(gcf, 'Name', [get(gcf, 'Name') ' - with amplitude overshoot-correction']);
    end
    
    hs(2) = subplot(2,2,2);
    bar(rout, h); xlabel('normalized breathing amplitude'); ylabel('counts');title('histogram for phase mapping');
    xlim([-1.1 1.1]);
    feqht = cumsumh/sumh*pi;
    hs(3) = subplot(2,2,3); plot(rout, [feqht, cos(feqht), sin(feqht)]); 
    xlabel('normalized breathing amplitude'); 
    title('equalized histogram bin amplitude -> phase transfer function (f_{eqht})');
    legend('f: normalized amplitude -> phase transfer function', 'cos(f)', 'sin(f)', ...
        'Location', 'NorthWest');
    
    %figure('Name', 'Histogram: Respiration phase estimation');
    hs(4) = subplot(2,2,4);
    plot(t, [normpulse*10, dnormpulse, (rphase-pi)]);
    legend('10*normalized breathing belt amplitude', '-1 = exhale, 1 = inhale', 'estimated respiratory phase');    
    ylim([-10.2 10.2]);
    title('Histogram-based respiration phase estimation');
    
    linkaxes(hs([1 4]), 'x');
    linkaxes(hs([2 3]), 'x');
else
    fh = [];
end
