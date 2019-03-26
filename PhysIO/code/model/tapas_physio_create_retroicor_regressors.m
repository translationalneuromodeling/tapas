function [cardiac_sess, respire_sess, mult_sess, ons_secs, order, verbose, ...
    c_sample_phase, r_sample_phase] ...
    = tapas_physio_create_retroicor_regressors(ons_secs, sqpar, order, verbose)
% calculation of regressors for physiological motion correction using RETROICOR (Glover, MRM, 2000)
%
% USAGE:
%   [cardiac_sess, respire_sess, mult_sess, verbose, c_sample_phase, r_sample_phase] ...
%        = tapas_physio_create_retroicor_regressors(ons_secs, sqpar, thresh, slicenum, order, verbose)
%
% NOTE: also updates order of models to 0, if some data does not exist
% (cardiac or respiratory)
%
% INPUT:
%   ons_secs    - onsets of all physlog events in seconds
%               .spulse     = onsets of slice scan acquisition
%               .cpulse     = onsets of cardiac R-wave peaks
%               .r          = time series of respiration
%               .svolpulse  = onsets of volume scan acquisition
%               .t          = time vector of logfile rows
%
%   sqpar       - sequence timing parameters
%           .Nslices        - number of slices per volume in fMRI scan
%           .NslicesPerBeat - usually equals Nslices, unless you trigger
%                             with the heart beat
%           .TR             - repetition time in seconds
%           .Ndummies       - number of dummy volumes
%           .Nscans         - number of full volumes saved (volumes in nifti file,
%                             usually rows in your design matrix)
%            onset_slice    - slice whose scan onset determines the adjustment of the
%                             regressor timing to a particular slice for the whole volume
%
%
% -------------------------------------------------------------------------
% Lars Kasper, March 2012
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


%% variable renaming
if ~exist('verbose', 'var')
    verbose.level = 1;
end

svolpulse       = ons_secs.svolpulse;
cpulse          = ons_secs.cpulse;
r               = ons_secs.r;
spulse          = ons_secs.spulse;
t               = ons_secs.t;

hasRespData = ~isempty(r);
hasPhaseData = isfield(ons_secs, 'c_sample_phase') && ~isempty(ons_secs.c_sample_phase);
hasCardiacData = hasPhaseData || ~isempty(cpulse);

if ~hasPhaseData
    
    % update model order, if resp/cardiac data do not exist
    if ~hasCardiacData
        order.c = 0;
    end
    
    if ~hasRespData
        order.r = 0;
    end
    
    if ~hasRespData || ~hasCardiacData
        order.cr = 0;
    end
    
    % compute phases from pulse data
    
    % compute differently, i.e. separate regressors for multiple slice
    % generation
    nSampleSlices = numel(sqpar.onset_slice);
    
    %parameters for resampling
    rsampint    = t(2)-t(1);
    
    %% Get phase, downsample and Fourier-expand
    sample_points   = tapas_physio_get_sample_points(ons_secs, sqpar);
    
    % cardiac phase estimation and Fourier expansion
    if (order.c || order.cr)
        
        [c_phase, verbose]    = ...
            tapas_physio_get_cardiac_phase(cpulse, spulse, verbose, svolpulse);
        c_sample_phase  = tapas_physio_downsample_phase(spulse, c_phase, sample_points, rsampint);
        cardiac_sess    = tapas_physio_get_fourier_expansion(c_sample_phase,order.c);
        
        cardiac_sess = tapas_physio_split_regressor_slices(cardiac_sess, ...
            nSampleSlices);
        
        
        
    else
        cardiac_sess = [];
        c_sample_phase = [];
    end
    
    % Respiratory phase estimation and Fourier expansion
    if (order.r || order.cr)
        
        fr = ons_secs.fr;
        
        if verbose.level >=3
            [r_phase, verbose.fig_handles(end+1)] = ...
                tapas_physio_get_respiratory_phase( ...
                fr,rsampint, verbose.level);
        else
            r_phase = tapas_physio_get_respiratory_phase(fr,rsampint, 0);
        end
        r_sample_phase  = tapas_physio_downsample_phase(t, r_phase, sample_points, rsampint);
        
        respire_sess    = tapas_physio_get_fourier_expansion(r_sample_phase,order.r);
        respire_sess = tapas_physio_split_regressor_slices(respire_sess, ...
            nSampleSlices);
        
    else
        respire_sess = [];
        r_sample_phase =[];
    end
    
else % compute Fourier expansion directly from cardiac/respiratory phases
    % select subset of slice-wise sampled phase for current onset_slice
    c_sample_phase = ons_secs.c_sample_phase;
    r_sample_phase = ons_secs.r_sample_phase;
    if (order.c || order.cr)
        cardiac_sess    = tapas_physio_get_fourier_expansion(...
            c_sample_phase, order.c);
        respire_sess    = tapas_physio_get_fourier_expansion(...
            r_sample_phase, order.r);
    else
        cardiac_sess = [];
        respire_sess = [];
    end
    
end


% Multiplicative terms as specified in Harvey et al., 2008
if order.cr && hasRespData && hasCardiacData
    crplus_sess = tapas_physio_get_fourier_expansion(c_sample_phase+r_sample_phase,order.cr);
    crdiff_sess = tapas_physio_get_fourier_expansion(c_sample_phase-r_sample_phase,order.cr);
    mult_sess = [crplus_sess crdiff_sess];
    mult_sess = tapas_physio_split_regressor_slices(mult_sess, ...
        nSampleSlices);
else
    mult_sess = [];
end

ons_secs.c_sample_phase = tapas_physio_split_regressor_slices(...
    c_sample_phase, nSampleSlices);
ons_secs.r_sample_phase =  tapas_physio_split_regressor_slices(...
    r_sample_phase, nSampleSlices);


%% plot cardiac & resp. regressors
if verbose.level >=2
    R = [cardiac_sess, respire_sess, mult_sess];
    
    hasCardiacData = ~isempty(ons_secs.c);
    hasRespData = ~isempty(ons_secs.r);
    
    verbose.fig_handles(end+1) = ...
        tapas_physio_plot_retroicor_regressors(R, order, hasCardiacData, ...
        hasRespData);
    
end
