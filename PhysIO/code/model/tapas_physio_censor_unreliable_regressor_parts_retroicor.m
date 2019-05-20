function [ons_secs, censored_cardiac_sess, censored_respire_sess, censored_mult_sess, verbose] = ...
    tapas_physio_censor_unreliable_regressor_parts_retroicor(...
    ons_secs, sqpar, cardiac_sess, respire_sess, mult_sess, verbose)
% Censors unreliable regressor parts (sets them to zero) for RETROICOR
% after downsampling censoring part to volume times
%
%   output = tapas_physio_censor_unreliable_regressor_parts_retroicor(input)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_censor_unreliable_regressor_parts_retroicor
%
%   See also

% Author:   Lars Kasper, after fruitful exchange with
%           [Daniel Hoffmann Ayala](https://github.com/DanielHoffmannAyala)
%
% Created: 2018-02-22
% Copyright (C) 2018 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.


%% Resample cardiac and respiratory flagged time series of reliability to contain only the time points used in the design matrix
doPlot = verbose.level >=2;

hasCardiacData      = ~isempty(cardiac_sess);
hasRespData         = ~isempty(respire_sess);
hasInteractionData  = ~isempty(mult_sess);

censored_cardiac_sess = [];
censored_respire_sess = [];
censored_mult_sess = [];


sample_points = tapas_physio_get_sample_points(ons_secs,sqpar);
t = ons_secs.t;

if hasCardiacData
    [censored_cardiac_sess, ons_secs.c_is_reliable_at_sample_points] = ...
        censor_unreliable_part(cardiac_sess, ons_secs.c_is_reliable, ...
        sample_points, t);
end

if hasRespData
    [censored_respire_sess, ons_secs.r_is_reliable_at_sample_points] = ...
        censor_unreliable_part(respire_sess, ons_secs.r_is_reliable, ...
        sample_points, t);
end

if hasInteractionData
    ons_secs.cr_is_reliable = ons_secs.c_is_reliable & ...
        ons_secs.r_is_reliable;
    [censored_mult_sess, ons_secs.cr_is_reliable_at_sample_points] = ...
        censor_unreliable_part(mult_sess, ons_secs.cr_is_reliable, ...
        sample_points, t);
end

if doPlot
    stringTitle = ['Model: Censoring of RETROICOR regressors in intervals of ' ...
        'unreliable physiological recordings'];
    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    set(gcf, 'Name', stringTitle);
    
    if hasCardiacData
        subplot(3,1,1);
        plot_censoring(censored_cardiac_sess, cardiac_sess, sample_points, t, ...
            ons_secs.c_is_reliable, ons_secs.c_is_reliable_at_sample_points, 'Cardiac', 'r')
    end
    
    if hasRespData
        subplot(3,1,2);
        plot_censoring(censored_respire_sess, respire_sess, sample_points, t, ...
            ons_secs.r_is_reliable, ons_secs.r_is_reliable_at_sample_points, 'Respiration', 'b')
    end
    
    if hasInteractionData
        subplot(3,1,3);
        plot_censoring(censored_mult_sess, mult_sess, sample_points, t, ...
            ons_secs.cr_is_reliable, ons_secs.cr_is_reliable_at_sample_points, 'Interaction', 'g')
        
    end
    
    xlabel('time (s)')
    tapas_physio_suptitle(stringTitle);
end

end

%% local function performing downsampling of reliability-vector mask and
%  censoring of regressor; avoids code duplication
function [censoredR, is_reliable_at_sample_points] = ...
    censor_unreliable_part(R, is_reliable, sample_points, t)

rsampint = t(2)-t(1);

% DOWNSAMPLE with default downsampling function
is_reliable_at_sample_points = tapas_physio_downsample_phase(t, ...
    is_reliable, sample_points, rsampint);

censoredR = R;
censoredR(~is_reliable_at_sample_points,:) = 0;

end


%% local function plotting censoring, plots in current axis
function  plot_censoring(censoredR, R, sample_points, t, ...
    is_reliable, is_reliable_at_sample_points, stringRegressor, col)
hp = plot(sample_points, R, [col '--']); hold all; hp = hp(1);
hp2 = plot(sample_points, censoredR, [col '-']);
hp = [hp, hp2(1)];
hp(3) = plot(t, is_reliable, 'c-');
hp(4) = stem(sample_points, is_reliable_at_sample_points, 'c');
ylim([-1.1 1.1]);
title(sprintf('Censored %s regressors', stringRegressor));
legend(hp, 'original regressors', 'censored regressors', ...
    'is reliable (=1) mask',...
    'is reliable at sampling points');
end