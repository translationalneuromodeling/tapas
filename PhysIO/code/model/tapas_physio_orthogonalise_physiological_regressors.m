function [R, verbose] = tapas_physio_orthogonalise_physiological_regressors(R, column_names, orthogonalise, verbose)
% orthogonalises (parts of) created physiological regressors
%
%   output = tapas_physio_orthogonalise_physiological_regressors(input)
% 
% Note: This is only necessary to ensure numerical stability for
% acuqisitions with cardiac triggering, where the cardiac phase
% coefficients are nearly constant throughout the volumes.
%
% IN
%   R               [Nscans, Nregressors]: Matrix of regressors
%   column_names    {Nregressors}: Cell array of regressor names
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
%             'RETROICOR'       - cardiac, resp and interaction (mult)
%                                 regressors are orthogonalised
%             'all'             - all regressors are orthogonalised to each other
%   verbose.level         0 = no output; 
%                   1 or other = plot design matrix before and after 
%                   orthogonalisation of physiological regressors and difference
% OUT
%
% EXAMPLE
%   tapas_physio_orthogonalise_physiological_regressors
%
%   See also

% Author: Lars Kasper
% Created: 2013-02-21
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

if isempty(R)
    return;
end

R_non_orth = R;
card_inds = strcmpi(column_names, 'RETROICOR (cardiac)');
resp_inds = strcmpi(column_names, 'RETROICOR (respiratory)');
mult_inds = strcmpi(column_names, 'RETROICOR (multiplicative)');
retr_inds = (card_inds | resp_inds | mult_inds);

switch lower(orthogonalise)
    case {'c', 'cardiac'}
        R(:, card_inds) = tapas_physio_scaleorthmean_regressors(R(:, card_inds));
        R(:, mult_inds) = tapas_physio_scaleorthmean_regressors(R(:, mult_inds));  % Lars: Why?
    case {'r', 'resp'}
        R(:, resp_inds) = tapas_physio_scaleorthmean_regressors(R(:, resp_inds));
    case {'mult'}
        R(:, mult_inds) = tapas_physio_scaleorthmean_regressors(R(:, mult_inds));
    case 'retroicor'
        R(:, retr_inds) = tapas_physio_scaleorthmean_regressors(R(:, retr_inds));
    case 'all'
        R = tapas_physio_scaleorthmean_regressors(R);
    case {'n', 'none'}
        % Easy!
    otherwise
        verbose = tapas_physio_log(...
            sprintf('Orthogonalisation of regressor set %s is not supported yet', ...
            orthogonalise), verbose, 2)
end


%% Plot orthogonalisation
if verbose.level

    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params(verbose);
    set(gcf, 'Name', 'Model: RETROICOR GLM regressors');
    
    switch lower(orthogonalise)
        case {'n', 'none'}
            imagesc(R); 
            title({'Model: Physiological regressor matrix for GLM', ...
                '- including input confound regressors -'});
            colormap gray;
            xlabel('Regressor'); ylabel('Scan volume');
            xticks(1:size(R, 2)); xticklabels(column_names); xtickangle(60);
            
        otherwise
            subplot(1,3,1); imagesc(R); title({'Model: Physiological regressor matrix for GLM'...
                ' - specified regressors orthogonalized - '}); 
            colormap gray; xlabel('regressor');ylabel('scan volume');
            subplot(1,3,2);
            imagesc(R_non_orth);
            title('non-orthogonalized regressors for GLM'); 
            colormap gray; xlabel('regressor');
            subplot(1,3,3);
            imagesc((R_non_orth-R).^2 );
            title({'squared differences of raw RETROICOR matrix to'...
                ' matrix with orthogonalized cardiac regressors'}); 
            colormap gray; xlabel('regressor'); colorbar;
    end
end
