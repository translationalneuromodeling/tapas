function [ obj ] = bold_grad_fd( obj, n )
% Calculate jacobian matrix of predicted fMRI BOLD signal with respect to
% DCM parameters using forward difference method.
% 
% This is a protected method of the tapas_Huge class. It cannot be called
% from outside the class.
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


% prediction error for current mean DCM parameter
mu_n = obj.posterior.mu_n(n,:);
data = obj.data(n);
obj.aux.epsilon{n} = obj.bold_gen(mu_n, data, obj.inputs(n), ...
    obj.options.hemo, obj.R, obj.L, obj.idx );

% calculate gradient
%   reserve memory
obj.aux.G{n} = zeros(numel(obj.data(n).bold), obj.idx.P_c + obj.idx.P_h);
%   set prediction at current parameter as base
data.bold =  data.bold - obj.aux.epsilon{n};
% forward difference
for p = 1:obj.idx.P_c + obj.idx.P_h    
    % parse parameters
    mu_n = obj.posterior.mu_n(n,:);
    mu_n(p) = mu_n(p) + obj.options.delta;
    delta = obj.bold_gen(mu_n, data, obj.inputs(n), obj.options.hemo,...
        obj.R, obj.L, obj.idx );
    % calculate difference
    obj.aux.G{n}(:,p) = - delta(:);
end
% normalize by step size
obj.aux.G{n} = obj.aux.G{n}/obj.options.delta;

end

