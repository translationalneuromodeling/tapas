function [cpulse, verbose] = tapas_physio_get_cardiac_pulses_manual_template(...
    c, t, thresh_cardiac_initial_cpulse_select, verbose)
% Detects R-peaks via matched-filter smoothing & peak detection using a
% manually defined QRS-wave (or R-peak environment)
%
%   [cpulse, verbose] = tapas_physio_get_cardiac_pulses_manual_template(...
%    c, t, thresh_min, dt120, verbose)
%
% IN
%   c               [nSamples, 1] raw pulse oximeter samples
%   t               [nSamples, 1] time vector corresponding to samples (un seconds)
%   thresh_cardiac_initial_cpulse_select   
%                   physio.thresh.cardiac.initial_cpulse_select-substructure
%                   with elements
%                   .method 'manual' or 'load'/'load_template'
%                           'manual' - select template manually
%                           'load'/'load_template' - load from template
%                   .min     threshold for correlation with QRS-wave to find cardiac pulses
%                   .file   variable saving an example cardiac QRS-wave to correlate with ECG time series
%   verbose         Substructure of Physio, holding verbose.level and
%                   verbose.fig_handles with plotted figure handles
%                   debugging plots for thresholding are only provided, if verbose.level >=2
%
% OUT
%
% EXAMPLE
%   tapas_physio_get_cardiac_pulses_manual_template
%
%   See also tapa_physio_new
%
% Author: Lars Kasper
% Created: 2012-02-20
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the physIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_get_cardiac_pulses_manual_template.m 524 2014-08-13 16:21:56Z kasperla $
if nargin < 5
    verbose.level = 0;
    verbose.fig_handles = [];
end

% manual peak selection, if no file selected and loading is
% specified

hasKrpeakLogfile = exist(thresh_cardiac_initial_cpulse_select.file,'file') || ...
    exist([thresh_cardiac_initial_cpulse_select.file '.mat'],'file');

% if no file exists, also do manual peak-find
doSelectTemplateManually = strcmpi(...
    thresh_cardiac_initial_cpulse_select.method, 'manual') || ~hasKrpeakLogfile;

if doSelectTemplateManually
    thresh_cardiac_initial_cpulse_select.kRpeak = [];
    hasECGMin = isfield(thresh_cardiac_initial_cpulse_select, 'min') && ~isempty(thresh_cardiac_initial_cpulse_select.min);
    if ~hasECGMin
        thresh_cardiac_initial_cpulse_select.min = 0.5;
    end
else
    fprintf('Loading %s\n', thresh_cardiac_initial_cpulse_select.file);
    ECGfile = load(thresh_cardiac_initial_cpulse_select.file);
    thresh_cardiac_initial_cpulse_select.min = ECGfile.ECG_min;
    thresh_cardiac_initial_cpulse_select.kRpeak = ECGfile.kRpeak;
end

inp_events = [];
ECG_min = thresh_cardiac_initial_cpulse_select.min;
kRpeak = thresh_cardiac_initial_cpulse_select.kRpeak;
if doSelectTemplateManually
    while ECG_min
        [cpulse, ECG_min_new, kRpeak] = tapas_physio_find_ecg_r_peaks(t,c, ECG_min, [], inp_events);
        fprintf('Press 0, then return, if right ECG peaks were found\n');
        ECG_min = input('otherwise type next numerical choice for ECG_min and continue the selection: ');
    end
else
    [cpulse, ECG_min_new, kRpeak] = tapas_physio_find_ecg_r_peaks(t,c, ECG_min, kRpeak, inp_events);
end
ECG_min = ECG_min_new;
cpulse = t(cpulse);

% save manually found peak parameters to file
if doSelectTemplateManually
    save(thresh_cardiac_initial_cpulse_select.file, 'ECG_min', 'kRpeak');
end