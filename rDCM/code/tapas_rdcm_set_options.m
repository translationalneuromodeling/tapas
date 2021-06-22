function [ options ] = tapas_rdcm_set_options(DCM, input_options, type)
% [ options ] = tapas_rdcm_set_options(DCM, input_options, type)
% 
% Sets options for the rDCM analysis to the default settings. Options can
% also be specified explicitly by the user.
% 
%   Input:
%   	DCM             - model structure
%       input_options   - estimation options (if empty, set to default)
%       type            - string which contains either 'r' for empirical data 
%                         or 's' for synthetic data
%
%   Output:
%       options         - estimation options
%
 
% ----------------------------------------------------------------------
% 
% Authors: Stefan Fraessle (stefanf@biomed.ee.ethz.ch), Ekaterina I. Lomakina
% 
% Copyright (C) 2016-2021 Translational Neuromodeling Unit
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


% set the data type (empirical or simulated data)
options.type = type;

% default settings
if strcmp(type, 's')
    options.SNR     = 3;
    options.y_dt    = 0.5;
    options.padding = 0;
else
    options.y_dt    = DCM.Y.dt;
    options.padding = 0;
end

% check for bilinear terms (not included in the current version of rDCM)
if ( isfield(DCM, 'b') && any(DCM.b(:)~=0) )
    options.bilinear = 1;
else 
    options.bilinear = 0;
end

% zero-padding of signal in frequency domain (default: none)
options.padding = 0;

% circular shift of the input (default: none)
options.u_shift = 0;

% filtering of signal in frequency domain (default: 1*STD)
options.filter_str          = 1;

% additional default settings for analysis
options.coef                = 1;
options.visualize           = 1;
options.compute_signal      = 1;

% create full covariance matrix (only recommended for small DCMs)
options.evalCp              = 0;


%% get specified settings from input_options

% overwrite default settings
if ~isempty(input_options)
    
    % specify all relevant fields
    names = fieldnames(options);
    
    % convolution is the only option which can be added in absence of the default value
    names{end+1} = 'h';
    
    % get the options from input_options
    for i = 1:length(names)
        if isfield(input_options,names(i))
            options.(names{i}) = input_options.(names{i});
        end
    end
end


% set settings that are only relevant for rDCM with sparsity constraints
if ( isfield(input_options,'p0_all') ),         options.p0_all = input_options.p0_all;                 end
if ( isfield(input_options,'iter') ),           options.iter = input_options.iter;                     end
if ( isfield(input_options,'restrictInputs') ), options.restrictInputs = input_options.restrictInputs; end
if ( isfield(input_options,'p0_inform') ),      options.p0_inform = input_options.p0_inform;           end


%% check options consistency

% get TR for real data
if strcmp(type, 'r')
    options.y_dt = DCM.Y.dt;
end

% sampling ratio between input and signal
r_dt = options.y_dt/DCM.U.dt;

% no up/subsampling needed
if ( r_dt == 1 )
    options.padding = 0;
end


%% compute fixed hemodynamic response function (HRF)

% compute only if HRF has not been pre-computed
if ( ~isfield(options,'h') || numel(options.h) ~= size(DCM.U.u,1) )
    options.DCM         = DCM;
    options.conv_dt     = DCM.U.dt;
    options.conv_length = size(DCM.U.u,1);
    options.conv_full   = 'true';
    options.h           = tapas_rdcm_get_convolution_bm(options);
end

end
