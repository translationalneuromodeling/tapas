%% [ J ] = tapas_huge_jacobian( fnFunction, argument, params, ~, varargin )
% 
% Calculates the jacobian matrix of a function via central differences
%
% INPUT:
%       fnFunction - handle to function
%       argument   - argument at which the jacobian is calculated
%       params     - struct containing parameters like step-size, etc.
%       ~          - a dummy input
%
% Optional:
%       varargin   - additional arguments of function
%
% OUTPUT: 
%       J          - the jacobian matrix
%

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2018 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <http://www.gnu.org/licenses/>.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is in an early stage of
% development. Considerable changes are planned for future releases. For
% support please refer to:
% https://github.com/translationalneuromodeling/tapas/issues
%
function [ J ] = tapas_huge_jacobian( fnFunction, argument, params, ~, varargin )

[dimIn,~] = size(argument);
dimOut = params.dimOut;

J = zeros(dimOut,dimIn);

if isfield(params,'stepSize')
    stepSize = params.stepSize;
else
    stepSize = 1e-3; % default step size
end
if isscalar(stepSize)
    stepSize = repmat(stepSize,dimIn,1);
end

for idxIn = 1:dimIn
    
    currentArgument = argument;
    currentArgument(idxIn) = currentArgument(idxIn) + stepSize(idxIn);
    valuePlus = fnFunction(currentArgument,varargin{:});
    
    currentArgument = argument;
    currentArgument(idxIn) = currentArgument(idxIn) - stepSize(idxIn);
    valueMinus = fnFunction(currentArgument,varargin{:});
    
    J(:,idxIn) = (valuePlus - valueMinus)/2/stepSize(idxIn);
        
end

% if requested, switch to denominator layout (gradient is a column vector)
% default: numerator layout (gradient is a row vector)
if isfield(params,'denominatorLayout')&&params.denominatorLayout
    J = J.';
end



end


