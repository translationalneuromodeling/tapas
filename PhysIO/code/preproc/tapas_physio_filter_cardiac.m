function [fc, verbose] = tapas_physio_filter_cardiac(t,c, options, verbose)
% Bandpass-filters cardiac data using butterworth/chebyshev filters
%
%    [fc, verbose] = tapas_physio_filter_cardiac(t,c,options, verbose)
%
% IN
%   t           [nSamples,1] time vector of cardiac time series
%   c           [nSamples, 1] cardiac time series
%   options     filter options
%               type    'butter' Butterworth, flat passband) or
%                       'cheby2' Chebychev Type II, for steep transitions
%                                between pass/stop band
%               passband [f_min, f_max] in Hz
%               stopband [f_min, f_max] in Hz
%   verbose     if .level >=2, output plot comparing filtered/unfiltered
%               time series
%
% OUT
%   fc          filtered time series
%   verbose     if plots were created, figure handles added here
%
% EXAMPLE
%   tapas_physio_filter_cardiac
%
%   See also tapas_physio_new

% Author:   Lars Kasper, based on code snippets at
%           https://ch.mathworks.com/matlabcentral/answers/327475-band-pass-butterworth-filter
% Created:  2019-07-02
% Copyright (C) 2019 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.

doFilter = options.include && ~isempty(options.passband);

if ~doFilter
    fc = c;
else
    % set default stopband 10% bigger than passband
    if isempty(options.stopband)
        widthPassBand = diff(options.passband);
        marginTransitionBand = 0.1*widthPassBand;
        options.stopband = [ options.passband(1)-marginTransitionBand, ...
            options.passband(2)+marginTransitionBand ];
    end
    
    Fsp = 1/(t(2) - t(1)); % sampling rate
    deltaF = 1/(t(end)-t(1));
    
    % filter boundaries check!
    options.passband(1) = max(2*deltaF, options.passband(1));
    options.passband(2) = min(Fsp-deltaF, options.passband(2));
    options.stopband(1) = max(deltaF, options.stopband(1));
    options.stopband(2) = min(Fsp, options.stopband(2));
    
    
    Fn = Fsp/2;
    Wp = options.passband/Fn;
    Ws = options.stopband/Fn;
    
    switch lower(options.type)
        case {'butter', 'butterworth'}
            
            [z,p,k]=butter(8,Wp,'bandpass');
            [sos,g]=zp2sos(z,p,k);
            
            %             Rp=1;
            %             Rs=25;
            %             [n,Wn] = buttord(Wp,Ws,Rp,Rs);
            %             [b,a]=butter(n,Wn);
            %             [sos,g]=tf2sos(b,a);
        case {'cheby2', 'chebychev'}
            Rp=10;
            Rs=30;
            [n,Ws] = cheb2ord(Wp,Ws,Rp,Rs);
            [z,p,k] = cheby2(n,Rs,Ws);
            [sos,g] = zp2sos(z,p,k);
    end
    fc=filtfilt(sos,g,c);
    
    
    %% plot filter response, but not for standalone (deployed) code or during
    % matlab-compilation, because fvtool is not part of compilation license
    if verbose.level >=3 && ~(isdeployed || ismcc)
        fvtool(sos,'Analysis','freq')
    end
    
    %% plot filtering results
    if verbose.level >=2
        stringTitle = 'Preproc: Bandpass-filtered Filtered Cardiac time series';
        verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
        set(gcf, 'Name', stringTitle);
        
        % plot raw/filtered time series
        subplot(2,1,1);
        plot(t, c); hold all;
        plot(t, fc);
        xlabel('t(s)');
        ylabel('Cardiac Wave Amplitude (a.u.)');
        legend('raw', sprintf('filtered (%s)', options.type));
        title(stringTitle);
        
        subplot(2,1,2);
        tapas_physio_plot_spectrum(t,[c fc], gca);
        legend('raw', sprintf('filtered (%s)', options.type));
    end
end
