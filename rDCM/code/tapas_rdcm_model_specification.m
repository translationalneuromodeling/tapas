function [ DCM ] = tapas_rdcm_model_specification(Y, U, args)
% [ DCM ] = tapas_rdcm_model_specification(Y, U, args)
% 
% Utilizes data and driving inputs to specify a whole-brain dynamic causal
% modeling (DCM) structure that can be utilized for inference with the
% regression DCM (rDCM) toolbox.
% 
%   Input:
%       Y               - data structure
%       Y.y             - data [NxR]
%                           N = number of datapoints
%                           R = number of regions
%       Y.dt            - repetition time [s]
%       Y.name          - region names (optional)
%                           default: region_1, ..., region_R
% 
%       U               - input structure 
%       U.u             - inputs [(16*N)xU] (if inputs are specified with dimension NxU, the function
%                                            will adapt inputs to match the correct microtime resolution)
%       U.name          - input names (optional)
%                           default: input_1, ..., input_U
% 
%       args            - arguments
%
%   Output:
%       DCM             - DCM structure
% 
% Note: Some fields are not (yet) used by the rDCM toolbox but are there
% for consistency with the original DCM framework (SPM) and/or because they
% might represent features that will be added in future releases.
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


% check if data and inputs are defined correctly
if ( ~isfield(Y,'y') || (~isempty(U) && ~isfield(U,'u')) )
    fprintf('\nERROR: Data (y) and/or inputs (u) not correctly specified! \n')
    fprintf('Please double-check... \n')
    DCM = [];
    return;
end


% specify connectivity matrices (just for order in structure)
DCM.a = [];
DCM.b = [];
DCM.c = [];
DCM.d = [];


% specify data
DCM.Y.y  = Y.y;
DCM.Y.dt = Y.dt;

% specify region names
if ( isfield(Y,'name') && iscell(Y.name) )
    DCM.Y.name = Y.name;
else
    for Nr_region = 1:size(Y.y,2)
        DCM.Y.name{Nr_region} = ['region_' num2str(Nr_region)];
    end
end


% specification of inputs
if ( ~isempty(U) )

    % check for dimensionality of inputs
    if ( size(U.u,1) == size(DCM.Y.y,1)*16 )
        u_temp = U.u;
    elseif ( size(U.u,1) == size(DCM.Y.y,1) )
        u_temp = zeros(size(U.u,1)*16,size(U.u,2));
        for Nr_input = 1:size(U.u,2)
            uu = U.u(:,Nr_input)';
            uu = repmat(uu,16,1);
            u_temp(:,Nr_input) = uu(:);
        end
    else
        fprintf('\nERROR: Dimensionality of data (y) and inputs (u) does not match! \n')
        fprintf('Please double-check... \n')
        DCM = [];
        return;
    end

    % specify inputs
    DCM.U.u = u_temp;

    % sampling rate of inputs
    DCM.U.dt = DCM.Y.dt/16;

    % specify input names
    if ( isfield(U,'name') && iscell(U.name) )
        DCM.U.name = U.name;
    else
        for Nr_input = 1:size(U.u,2)
            DCM.U.name{Nr_input} = ['input_' num2str(Nr_input)];
        end
    end
    
else
    
    % make user aware that empty input argument is interpreted as 
    % resting-state fMRI data. For this case, no field U is defined as 
    % this will be handled automatically by "tapas_rdcm_estimate.m"
    fprintf('\nNOTE: No inputs specified! Assuming resting-state model... \n')
    
end


% specify number of datapoints (per regions)
DCM.v = size(DCM.Y.y,1);

% specify number of regions
DCM.n = size(DCM.Y.y,2);


% task based or resting state
if ( ~isempty(U) )

    % specify connectivity matrices (default: full connectivity and input)
    DCM.a = ones(size(Y.y,2));
    DCM.b = zeros(size(Y.y,2),size(Y.y,2),size(U.u,2));
    DCM.c = ones(size(Y.y,2),size(U.u,2));
    DCM.d = zeros(size(Y.y,2),size(Y.y,2),0);


    % overwrite default connectivity matrices
    if ( ~isempty(args) )

        % overwrite A-matrix
        if ( isfield(args,'a') )
            DCM.a = args.a;
        end

        % overwrite C-matrix
        if ( isfield(args,'c') )
            DCM.c = args.c;
        end
    end
    
else
    
    % specify connectivity matrices; sets dummy B, C, and D matrices as
    % these will be handled automatically by "tapas_rdcm_estimate.m"
    DCM.a = ones(size(Y.y,2));
    DCM.b = zeros(size(Y.y,2),size(Y.y,2),0);
    DCM.c = zeros(size(Y.y,2),0);
    DCM.d = zeros(size(Y.y,2),size(Y.y,2),0);


    % overwrite default connectivity matrices
    if ( ~isempty(args) )

        % overwrite A-matrix
        if ( isfield(args,'a') )
            DCM.a = args.a;
        end
    end
    
end


% specify delays (not used so far | might be added later)
DCM.delays = (DCM.Y.dt/2) * ones(DCM.v,1);

% specify options (not used so far | might be added later)
DCM.options.nonlinear   = 0;
DCM.options.two_state   = 0;
DCM.options.stochastic	= 0;
DCM.options.centre      = 0;
DCM.options.endogenous	= 0;

end
