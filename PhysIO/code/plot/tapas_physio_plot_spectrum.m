function [fh, f, P1] = tapas_physio_plot_spectrum(t, y, hAx)
% Plots one-sided amplitude spectrum of a time series, following Matlab
% help examples
%
%   [fh, f, P1] = tapas_physio_plot_spectrum(t, y)
%
% IN
%   t   [nSamples,1]    time vector
%   y   [nSamples,nTimeseries]  different signal time series, one per
%                               columns
%   hAx  axis handle for plot (optional, otherwise new figure is created)
%
% OUT
%   fh  figure handle
%   f   [nSamples/2+1,1] frequency vector (x axis)
%   P1  [nSamples/2+1,1] one-sided amplitude spectrum
%
% EXAMPLE
%   tapas_physio_plot_spectrum
%
%   See also

% Author:   Lars Kasper
% Created:  2019-07-03
% Copyright (C) 2019 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.

if nargin < 3
    fh = tapas_physio_get_default_fig_params();
else
    axes(hAx);
    fh = gcf;
end

% freq vector for time
L=size(y,1);
Fs = 1/(t(2)-t(1));
f = Fs*(0:(L/2))/L;

for iCols = 1:size(y,2)
    Y=fft(y(:,iCols));
    
    % single sided spectrum
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    plot(f,P1); hold all
end
stringTitle = 'Preproc: Single-Sided Amplitude Spectrum';
title(stringTitle);
xlabel('f (Hz)')
ylabel('|P1(f)|')

if nargin < 3 % update name of figure only for newly created one
    set(gcf, 'Name', stringTitle);
end