function [cardiac_sess, respire_sess, mult_sess, verbose, c_sample_phase, r_sample_phase] ...
    = tapas_physio_create_retroicor_regressors(ons_secs, sqpar, thresh, order, verbose) 
% calculation of regressors for physiological motion correction using RETROICOR (Glover, MRM, 2000)
%
% USAGE:
%   [cardiac_sess, respire_sess, mult_sess, verbose, c_sample_phase, r_sample_phase] ...
%        = tapas_physio_create_retroicor_regressors(ons_secs, sqpar, thresh, slicenum, order, verbose) 
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
%   thresh
%           .resp_max       -  if set, all peaks above that breathing belt amplitude 
%                              are ignored for respiratory phase evaluation
%
% -------------------------------------------------------------------------
% Lars Kasper, March 2012
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_create_retroicor_regressors.m 235 2013-08-19 16:28:07Z kasperla $

%% variable renaming
if ~exist('verbose', 'var')
    verbose.level = 1;
end
svolpulse       = ons_secs.svolpulse;
cpulse          = ons_secs.cpulse;
r               = ons_secs.r;
spulse          = ons_secs.spulse;
t               = ons_secs.t;
slicenum        = sqpar.onset_slice;

%parameters for resampling
rsampint    = t(2)-t(1);

%% Get phase, downsample and Fourier-expand
sample_points   = tapas_physio_get_sample_points(ons_secs, sqpar, slicenum);

if order.c
    if verbose.level >= 3
    [c_phase, verbose.fig_handles(end+1)]    = ...
        tapas_physio_get_cardiac_phase(cpulse, spulse, verbose.level, svolpulse);
    else
        c_phase    = tapas_physio_get_cardiac_phase(cpulse, spulse, 0, svolpulse);
    end
    c_sample_phase  = tapas_physio_downsample_phase(spulse, c_phase, sample_points, rsampint);
    cardiac_sess    = tapas_physio_get_fourier_expansion(c_sample_phase,order.c);
else
    cardiac_sess = [];
    c_sample_phase = [];
end

if order.r
    fr = tapas_physio_filter_respiratory(r,rsampint);
    
    if verbose.level >=3
        [r_phase, verbose.fig_handles(end+1)] = ...
            tapas_physio_get_respiratory_phase( ...
                fr,rsampint, verbose.level, thresh);
    else
        r_phase = tapas_physio_get_respiratory_phase(fr,rsampint, 0, thresh);
    end
    r_sample_phase  = tapas_physio_downsample_phase(t, r_phase, sample_points, rsampint);
    respire_sess    = tapas_physio_get_fourier_expansion(r_sample_phase,order.r);
else
    respire_sess = [];
    r_sample_phase =[];
end

% Multiplicative terms as specified in Harvey et al., 2008
if order.cr
    crplus_sess = tapas_physio_get_fourier_expansion(c_sample_phase+r_sample_phase,order.cr);
    crdiff_sess = tapas_physio_get_fourier_expansion(c_sample_phase-r_sample_phase,order.cr);
    mult_sess = [crplus_sess crdiff_sess];
else
    mult_sess = [];
end

subj='';
%% plot cardiac & resp. regressors
if verbose.level >=2
    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    
    set(gcf,'Name','RETROICOR timecourse physiological regressors');
    if order.cr
        Nsubs=3;
        ax(3) = subplot(Nsubs,1,3);
        plot([mult_sess+repmat(1:size(mult_sess,2),length(mult_sess),1)]);
        xlabel('scans');title([subj ' , RETROICOR multiplicative cardiac x respiratory regressors, vertical shift for visibility'])
    else
        Nsubs=2;
    end
    ax(1) = subplot(Nsubs,1,1);
    plot([cardiac_sess+repmat(1:size(cardiac_sess,2),length(cardiac_sess),1)]);
    xlabel('scans');title([subj ' , RETROICOR cardiac regressors, vertical shift for visibility'])
    
    ax(2) = subplot(Nsubs,1,2);
    plot([respire_sess+repmat(1:size(respire_sess,2),length(respire_sess),1)]);
    xlabel('scans');title([subj ' , RETROICOR respiratory regressors, vertical shift for visibility'])
    if ~(isempty(cardiac_sess) || isempty(respire_sess)), linkaxes(ax,'x'); end
end
