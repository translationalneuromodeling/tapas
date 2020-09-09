function [ epsilon ] = bold_gen( obj, theta, data, inputs, hemo, R, L, idx )
% Generate predicted fMRI BOLD time series..
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


% transform DCM parameters
if isstruct(theta)
    A = theta.A;
    B = theta.B;
    C = theta.C;
    D = theta.D;
    tau = theta.transit;
    kappa = theta.decay;
    epsilon = theta.epsilon;
else
    [A, B, C, D, tau, kappa, epsilon] = obj.theta2abcd( theta, idx, R, L );
end

% transform C matrix
if obj.options.nvp.transforminput
    C(logical(obj.dcm.c(:)))=.5*exp(C(logical(obj.dcm.c(:))));
end
% baseline for self-connections
A = A + obj.const.baseSc*eye(obj.R);


% generate BOLD response
pred = tapas_huge_bold( A, B, C, D, tau, kappa, epsilon, R, inputs.u, L, ...
    hemo.E_0, hemo.r_0, hemo.V_0, hemo.vartheta_0, hemo.alpha, hemo.gamma, ...
    data.tr, data.te, inputs.dt);

% remove 1st-level confounds
if ~isempty(data.res)
    pred = data.res*pred;
end

epsilon = data.bold - pred;

end

