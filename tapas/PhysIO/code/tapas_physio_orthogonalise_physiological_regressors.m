function [R, verbose] = tapas_physio_orthogonalise_physiological_regressors(cardiac_sess, respire_sess, mult_sess, input_R, orthogonalise, verbose)
% orthogonalises (parts of) created physiological regressors
%
%   output = tapas_physio_orthogonalise_physiological_regressors(input)
% 
% Note: This is only necessary to ensure numerical stability for
% acuqisitions with cardiac triggering, where the cardiac phase
% coefficients are nearly constant throughout the volumes.
%
% IN
%   cardiac_sess    [Nscans, order.c x 2] regressors of cardiac phase
%                   expansion
%   respire_sess    [Nscans, order.r x 2] regressors of respiratory phase
%                   expansion
%   mult_sess       [Nscans, order.cr x 2] regressors of cardiac X respiratory phase
%                   interaction expansion
%   input_R         other confound regressors (e.g. realignment parameters)
%   orthogonalise
%           - string indicating which regressors shall be
%             orthogonalised; mainly needed, if
%           acquisition was triggered to heartbeat (set to 'cardiac') OR
%           if session mean shall be evaluated (e.g. SFNR-studies, set to
%           'all')
%             'n' or 'none'     - no orthogonalisation is performed
%             'c' or 'cardiac'  - only cardiac regressors are orthogonalised
%             'r' or 'resp'     - only respiration regressors are orthogonalised
%             'mult'            - only multiplicative regressors are orthogonalised
%             'all'             - all physiological regressors are
%                                 orthogonalised to each other
%   verbose.level         0 = no output; 
%                   1 or other = plot design matrix before and after 
%                   orthogonalisation of physiological regressors and difference
% OUT
%
% EXAMPLE
%   tapas_physio_orthogonalise_physiological_regressors
%
%   See also
%
% Author: Lars Kasper
% Created: 2013-02-21
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_orthogonalise_physiological_regressors.m 408 2014-01-20 00:25:56Z kasperla $
R_non_orth = [cardiac_sess, respire_sess, mult_sess input_R];

if isempty(R_non_orth)
    R = [];
    return;
end

switch lower(orthogonalise)
    case {'c', 'cardiac'}
        R = [tapas_physio_scaleorthmean_regressors(cardiac_sess), respire_sess, tapas_physio_scaleorthmean_regressors(mult_sess) input_R];
    case {'r', 'resp'}
        R = [cardiac_sess, tapas_physio_scaleorthmean_regressors(respire_sess) mult_sess input_R];
    case {'mult'}
        R = [cardiac_sess, respire_sess, tapas_physio_scaleorthmean_regressors(mult_sess) input_R];
    case 'all'
        R = [tapas_physio_scaleorthmean_regressors([cardiac_sess, respire_sess, mult_sess]), input_R];
    case {'n', 'none'}
        R = [cardiac_sess, respire_sess, mult_sess input_R];
end


%% Plot orthogonalisation
if verbose.level
    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    set(gcf, 'Name', 'RETROICOR GLM regressors');
    subplot(1,3,1); imagesc(R); title({'physiological regressors matrix for GLM'...
        ' - specified regressors orthogonalized - '}); colormap gray; xlabel('regressor');ylabel('scan volume');
    subplot(1,3,2);
    imagesc(R_non_orth);title('non-orthogonalized regressors for GLM'); colormap gray; xlabel('regressor');
    subplot(1,3,3);
    imagesc((R_non_orth-R).^2 );title({'squared differences of raw RETROICOR matrix to'...
        ' matrix with orthogonalized cardiac regressors'}); colormap gray; xlabel('regressor'); colorbar;
end
