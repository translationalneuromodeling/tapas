function [output, options] = tapas_rdcm_estimate(DCM, type, options, methods)
% [output, options] = tapas_rdcm_estimate(DCM, type, options, methods)
% 
% Main analysis function, which calls the subfunctions necessary to run a
% regression DCM (rDCM) analysis.
% 
%   Input:
%   	DCM             - either model structure or a filename
%       type            - 'r' for empirical data or 's' for
%                         simulated data
%       options         - estimation options (if empty would be set to default)
%       methods         - (1) rDCM (original) or (2) rDCM with sparsity
%
%   Output:
%       output          - output structure
%       options         - estimation options
% 
%   Reference:
%       Frässle, S., Lomakina, E.I., Razi, A., Friston, K.J., Buhmann, J.M., 
%       Stephan, K.E., 2017. Regression DCM for fMRI. NeuroImage 155, 406-421.
%       https://doi.org/10.1016/j.neuroimage.2017.02.090
% 
%       Frässle, S., Lomakina, E.I., Kasper, L., Manjaly Z.M., Leff, A., 
%       Pruessmann, K.P., Buhmann, J.M., Stephan, K.E., 2018. A generative 
%       model of whole-brain effective connectivity. NeuroImage 179, 505-529.
%       https://doi.org/10.1016/j.neuroimage.2018.05.058
%
 
% ----------------------------------------------------------------------
% 
% Authors: Stefan Fraessle (stefanf@biomed.ee.ethz.ch), Ekaterina I. Lomakina
% 
% Copyright (C) 2016-2018 Translational Neuromodeling Unit
%                         Institute for Biomedical Engineering
%                         University of Zurich & ETH Zurich
%
% This file is part of the TAPAS rDCM Toolbox, which is released under the 
% terms of the GNU General Public License (GPL), version 3.0 or later. You
% can redistribute and/or modify the code under the terms of the GPL. For
% further see COPYING or <http://www.gnu.org/licenses/>.
% 
% Please note that this toolbox is in an early stage of development. Changes 
% are likely to occur in future releases.
% 
% ----------------------------------------------------------------------


% store the seed of the RNG
rngSeed = rng();


% display
fprintf('\n========================================================\n')
fprintf('Regression dynamic causal modeling (rDCM) \n')
fprintf('========================================================\n\n')


% load the DCM
if ~isstruct(DCM)
    DCM = load(DCM);
end


% create the options file
options = tapas_rdcm_set_options(DCM, options, type);


% create the regressors
[X, Y, DCM, args] = tapas_rdcm_create_regressors(DCM, options);


% display start
fprintf('Run model inversion\n')


% evaluate results (model inversion)
if ( methods == 1 )
    
    % rDCM (original)
    output = tapas_rdcm_ridge(DCM, X, Y, args ); 

elseif ( methods == 2 )
    
    % specify the default grid for optimizing sparsity hyperparameter p0
    if ( ~isfield(options,'p0_all') )
        options.p0_all	= 0.05:0.05:0.95;
    end
    
    % result array for all p0
    output_all  = cell(1,length(options.p0_all));
    F_all       = NaN(1,length(options.p0_all));
    
    % for command output
    reverseStr = '';
    
    % prune driving inputs or not
    if ( isfield(options,'restrictInputs') ), args.restrictInputs = options.restrictInputs; end
    
    % iterate over p0 values
    for p0_counter = 1:length(options.p0_all)
        
        % specify p0
        args.p0_temp = options.p0_all(p0_counter);
        
        % specify the number of permutations (per brain region)
        if ( isfield(options,'iter') ), args.iter = options.iter; end
        
        % output progress
        msg = sprintf('Processing p0: %d/%d', p0_counter, length(options.p0_all));
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
        
        % rDCM (with sparsity constraints)
         output_temp            = tapas_rdcm_sparse(DCM, X, Y, args);
         output_all{p0_counter} = output_temp{1};
        
        % get the negative free energy
        F_all(p0_counter) = output_all{p0_counter}.logF;
        
    end
    
    % find optimal hyperparameter settings (p0)
    [~, F_max_ind] = max(F_all);
    
    % asign results
    output = output_all{F_max_ind};
    
end

% display finalizing
fprintf('\nFinalize results\n')

% evaluate statistics and predicted signal
output = tapas_rdcm_compute_statistics(DCM, output, options);

% store the random number seed and the version number
output.rngSeed = rngSeed;
output.ver     = '2018_v01';

end
