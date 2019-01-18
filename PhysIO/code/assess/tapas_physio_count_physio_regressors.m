function nPhysioRegressors = tapas_physio_count_physio_regressors(physio)
% Returns number of physiological regressors created, given model
% specification; 
% NOTE: only reproducible numbers (data-independent) are
% returned, i.e. session-specific movement spikes and %-variance explained
% PCA-components are not included
%
% IN
%   physio  physio-structure, See also tapas_physio_new
%
% OUT
% nPhysioRegressors     number of physiological regressors, e.g. motion,
%                       retroicor, noise_rois
%                       but: ignores
%
% EXAMPLE
%   tapas_physio_report_contrasts
%
%   See also

% Author: Lars Kasper
% Created: 2014-10-16
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.



model = physio.model;

nPhysioRegressors = 0;

if model.hrv.include
    nPhysioRegressors = nPhysioRegressors + numel(model.hrv.delays);
end

if model.rvt.include
    nPhysioRegressors = nPhysioRegressors + numel(model.rvt.delays);
end


if model.retroicor.include
    order = model.retroicor.order;
    nPhysioRegressors = nPhysioRegressors + ...
        2*order.c + ...
        2*order.r + ...
        4* order.cr;
end

if model.noise_rois.include
    % TODO: what if number of components implicit?...shall we save this?
    nPhysioRegressors = nPhysioRegressors + ...
        numel(model.noise_rois.roi_files)*ceil(model.noise_rois.n_components+1); % + 1 for mean
end

if model.movement.include
    % TODO: what about variable regressors, that should not be
    % concatenated, e.g. movement outlier censoring
    nPhysioRegressors = nPhysioRegressors + model.movement.order;
end

end
