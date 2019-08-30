function [ obj ] = remove( obj, idx )
% Remove data (fMRI time series, confounds, DCM network structure, ... )
% and estimation results from HUGE object.
%
% INPUTS:
%   obj - A tapas_Huge object.
% 
% OPTIONAL INPUTS:
%   idx - Only remove data of the subjects indicated in 'idx'. 'idx' must
%         be a vector containing numeric or logical array indices.
% 
% OUTPUTS:
%   obj - A tapas_Huge object with data and results removed.
%
% EXAMPLES:
%   [obj] = REMOVE(obj)    Remove results and data of all subjects.
% 
%   [obj] = REMOVE(obj,1:5)    Remove results and data for first 5
%                              subjects. 
% 
%   [obj] = REMOVE(obj,'all') is the same as [obj] = REMOVE(obj)
% 
%   [obj] = REMOVE(obj,0) is the same as [obj] = REMOVE(obj)
% 
% See also tapas_Huge.IMPORT
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


%% check input
if nargin < 2 || (ischar(idx) && strcmpi(idx, 'all'))
    idx = 0;
end

if obj.N == 0 || isempty(idx)
    return
end

if isscalar(idx) && idx == 0
    idx = 1:obj.N;
end

assert((all(idx >= 1) && all(rem(idx,1) == 0)) || islogical(idx),...
    'TAPAS:HUGE:invalidIndex',...
    'idx must be real positive integers or logicals');

assert(max(idx) <= obj.N, 'TAPAS:HUGE:invalidIndex',...
    'idx exceeds number of subjects');

if islogical(idx)
    assert(numel(idx) == obj.N, 'TAPAS:HUGE:invalidIndex',...
        'idx must have length %u', obj.N);
end

%% remove data
obj.data(idx)   = [];
obj.inputs(idx) = [];

if isnumeric(obj.options.nvp.startingvaluedcm) && ...
        ~isempty(obj.options.nvp.startingvaluedcm)
    obj.options.nvp.startingvaluedcm(idx,:) = [];
end

if ~isempty(obj.model)
    obj.model.d(idx, :) = [];
    obj.model.theta_c(idx, :) = [];
    obj.model.theta_h(idx, :) = [];
    obj.model.lambda(idx, :) = [];
    obj.model.pi = sum(obj.model.d)/sum(obj.model.d(:));
end

obj.N = numel(obj.data);
if ~obj.N
    obj = reset_properties(obj);
end

% reset prior, posterior and trace
obj.prior = [];
obj.posterior = [];
obj.trace = [];

end


function [ obj ] = reset_properties( obj )

tmp = obj.default_options();
obj.options.nvp.priorclustermean        = tmp.nvp.priorclustermean;
obj.options.nvp.priorclustervariance    = tmp.nvp.priorclustervariance;
obj.options.nvp.priordegree             = tmp.nvp.priordegree;
obj.options.nvp.priorvarianceratio      = tmp.nvp.priorvarianceratio;
obj.options.nvp.startingvaluedcm        = tmp.nvp.startingvaluedcm;
obj.options.nvp.startingvaluegmm        = tmp.nvp.startingvaluegmm;
obj.options.nvp.confoundsvariant        = tmp.nvp.confoundsvariant;

obj.R = 0;
obj.L = 0;
obj.M = 0;
obj.dcm = [];
obj.labels = struct();
obj.model = [];    
obj.idx = struct();

end



