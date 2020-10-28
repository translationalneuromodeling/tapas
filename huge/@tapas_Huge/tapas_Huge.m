classdef tapas_Huge
%% tapas_Huge Hierarchical Unsupervised Generative Embedding
%   This class implements the HUGE model. HUGE stands for Hierarchical
%   Unsupervised Generative Embedding. It is a generative model for
%   (task-based) fMRI data from heterogeneous cohorts. For more details on
%   the theory behind the HUGE model, see:
%
%   Yao Y, Raman SS, Schiek M, Leff A, Frässle S, Stephan KE (2018).
%   Variational Bayesian Inversion for Hierarchical Unsupervised Generative
%   Embedding (HUGE). NeuroImage, 179: 604-619
%   https://doi.org/10.1016/j.neuroimage.2018.06.073
% 
%   This class stores data, options and results as properties; while the
%   main functionalities are provided by the class methods. For a more
%   detailed documentation, see the user manual (tapas_huge_manual.pdf).
%   For a quickstart guide, run the tutorial script (tapas_huge_demo.mlx).
% 
%   TAPAS_HUGE properties:
%       K         - Number of clusters.
%       tag       - Model description.
%       L         - Number of experimental inputs.
%       M         - Number of confounds.
%       N         - Number of subjects.
%       R         - Number of regions.
%       idx       - DCM parameter indices.
%       dcm       - DCM network structure in SPM's DCM format.
%       inputs    - Experimental inputs and sampling time interval
%       data      - fMRI BOLD time series and related data.
%       labels    - Region and input labels.
%       options   - Model options.
%       prior     - Parameters of prior distribution.
%       posterior - Parameters of posterior distribution.
%       trace     - Convergence diagnostics.
%       aux       - Auxiliary variables.
%       model     - Ground truth parameters used to generate the data when
%                   model contains synthetic data. 
%       const     - Constants of the model.
%       version   - Toolbox version.
% 
%   TAPAS_HUGE methods:
%       import   - Import fMRI data. Overwrite previous data and results.
%       remove   - Remove data and results.
%       export   - Export data and results to SPM's DCM format.
%       estimate - Fit model to data.
%       simulate - generate synthetic dataset.
%       plot     - Plot results.
%       save     - Save object properties to file.
%         
% 
%   See also: TAPAS_HUGE_DEMO
% 

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2019 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <https://www.gnu.org/licenses/>.
% 
% This software is provided "as is", without warranty of any kind, express
% or implied, including, but not limited to the warranties of
% merchantability, fitness for a particular purpose and non-infringement.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is under active development.
% Considerable changes may occur in future releases. For support please
% refer to:
% https://github.com/translationalneuromodeling/tapas/issues
% 

    properties (Access = public)
        K = 1 % number of clusters
        tag = '' % description
    end
    
    properties (SetAccess = protected)
        L = 0 % Number of experimental inputs.
        M = 0 % Number of confounds.
        N = 0 % Number of subjects.
        R = 0 % Number of regions.
        idx = struct( ) % DCM parameter indices.
        
        dcm = [] % DCM network structure in SPM's DCM format.
        inputs = struct('u',  {}, ... % Experimental inputs.
                        'dt', {}) % sampling time interval of inputs.
        data = struct( ... % fMRI BOLD time series and related data.
            'bold',      {}, ... % BOLD time series
            'te',        {}, ... % echo time TE
            'tr',        {}, ... % repetition time TR
            'X0',        {}, ... % subject-level confounds
            'res',       {}, ... % residual forming matrix of X0
            'confounds', {}, ... % group-level confounds
            'spm',       {})     % posterior from SPM
        labels = struct( ) % Region and input labels.
        
        options   % Model options.
        prior     % Parameters of prior distribution.
        posterior % Parameters of posterior distribution.
        trace     % Convergence diagnostics.
        aux       % Auxiliary variables.
        
        model % Ground truth parameters if model contains synthetic data.
        
        const = struct( ... % Model constants
            'nKmeans', 10, ... % number of repetitions for kmeans
            'minLogVar', -5, ... % minimum log-variance BOLD
            'mhRate', .4, ... % target acceptance rate
            'mhAdapt', 3e3, ... % interval for adapting step sizes
            'mhTrans', 2^10, ... % sample size for adapting transform
            'mhReg',  9, ... % regularizer for adapting step sizes
            'nPsrf', 1e5, ... % rate for convergence monitoring via PSRF
            'baseSc', -.5) % baseline self-connection

        version = '2020-09'; % Toolbox version
    end


    % constructor and property access
    methods
        % constructor
        function obj = tapas_Huge( varargin )
% Create instance of tapas_Huge class.
%      
% INPUTS:     
%   This function accepts optional name-value pair arguments. For a list of
%   valid name-value pairs, see the user manual or type 'help
%   tapas_huge_property_names'.   
%         
% OUTPUTS:
%   obj - A tapas_Huge object.        
%         
% EXAMPLES:
%   [obj] = TAPAS_HUGE()    Create empty TAPAS_HUGE object.
%
%   [obj] = TAPAS_HUGE('tag','my model')    Create empty TAPAS_HUGE object
%       and add a short description.
%
%   [obj] = TAPAS_HUGE('Dcm',dcms)    Create TAPAS_HUGE object and import
%       fMRI time series data. 
%
%   [obj] = TAPAS_HUGE('K',2,'Dcm',dcms)    Create empty TAPAS_HUGE object
%       and set the number of clusters to 2. 
%
%   [obj] = TAPAS_HUGE('Dcm',dcms,'Verbose',true)    Create TAPAS_HUGE
%       object and activate command line output.
%
%   


            % set default options
            obj.options = obj.default_options( );
                        
            if nargin == 1
                %%% TODO build from posterior
            elseif nargin > 1
                % key value pairs
                obj = obj.optional_inputs(varargin{:});
            end            
        end % constructor end
  
        function obj = set.K(obj, val)
            assert(isscalar(val) && isnumeric(val) && val >= 1 && ...
                rem(val,1) == 0, 'TAPAS:HUGE:invalidK',...
                'Number of clusters K must be integer and larger or equal 1.');
            obj.K = val;
        end
        
        function obj = set.tag(obj, val)
            assert(ischar(val), 'TAPAS:HUGE:invalidTag', ...
                'Tag must be character array.');
            obj.tag = val;
        end
    end
    
    methods (Access = public)
        
        % import subject data
        [ obj ] = import( obj, listDcms, listConfounds, omit, bAppend )
        % remove subject data
        [ obj ] = remove( obj, idx )
        % export subject data to SPM format
        [ listDcms, listConfounds ] = export( obj );
        
        % invert the model
        [ obj ] = estimate( obj, varargin )
        
        % generate synthetic dataset
        [ obj ] = simulate( obj, clusters, sizes, varargin )
        
        % plot posterior
        [ fHdl ] = plot( obj, subjects )
        
        % save object properties to disk
        [ ] = save( filename, obj, varargin )
        
    end
    methods (Static, Access = public)       
        
        % unit testing
        [ ] = test( )
        
    end
    methods (Access = protected)
        
        % process name-value-pair inputs
        [ obj ] = optional_inputs( obj, varargin )
        % set default options
        [ obj ] = default_options( obj )
        
        % generate BOLD signal from DCM parameters
        [ epsilon ] = bold_gen( obj, theta, data, inputs, hemo, R, L, idx )
        % derivative of BOLD signal wrt mean DCM parameters
        [ obj ] = bold_grad_cd( obj, n ) % central difference
        [ obj ] = bold_grad_fd( obj, n ) % forward difference
        
        % VB inversion
        [ obj ] = vb_invert( obj )
        % VB negative free energy
        [ nfe ] = vb_nfe( obj, ns )
        % VB initialization
        [ obj ] = vb_init( obj )
        
    end
    methods (Static, Access = protected)
        
        % de-multiplex parameter vector
        [A, B, C, D, tau, kappa, epsilon] = theta2abcd(theta, idx, R, L)
        % generate labels for axis ticks
        [ tickLabels ] = parse_labels( dcm, labels, idx )
    end
    
end