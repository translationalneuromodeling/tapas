function [y] = tapas_mpdcm_fmri_int_host(u, theta, ptheta, sloppy)
%% Integrates the system of differential equations specified by the input 
% in the host.
%
% Input:
% u         -- Cell array of experimental input
% theta     -- Cell array of model parameteres
% ptheta    -- Structure. Model priors or constants
% sloppy    -- Scalar. If true don't check the input. 
%
% Ouput:
% y         -- Cell array of predicted signals.
%
% If the input is not compliant, it's very likely that a segmentation fault
% happens and that matlab closes. It should only be used once the input has
% been check at least once and changes to them are done via well tested
% functions.
%

% aponteeduardo@gmail.com
%
% Author: Eduardo Aponte, TNU, UZH & ETHZ - 2015
% Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
%
% Licensed under GNU General Public License 3.0 or later.
% Some rights reserved. See COPYING, AUTHORS.
%
% Revision log:
%
%

if nargin < 4
    sloppy = 0;
end


if ~sloppy
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);  
end

if isfield(ptheta, 'integ')
    switch ptheta.integ
    case 'euler'
        integ = @(u, theta, ptheta) c_mpdcm_fmri_euler_host(u, theta, ptheta);
    case 'rk4'
        integ = @(u, theta, ptheta) c_mpdcm_fmri_rk4_host(u, theta, ptheta);
    otherwise
        error('mpdcm:fmri:int:input', 'Unknown integrator.');
    end
else
    error('mpdcm:fmri:int:input', 'Integrator must be specified.');
end

hps = tapas_mpdcm_fmri_get_hempars();

for i = 1:numel(theta)
    theta{i}.C = theta{i}.C/16;
    theta{i}.D = theta{i}.D;
    [k1, k2, k3] = tapas_mpdcm_fmri_k(theta{i});
    theta{i}.k1 = k1;
    theta{i}.k2 = k2;
    theta{i}.k3 = k3;
    % Change the parametrization
    theta{i}.K = hps.K * exp(theta{i}.K);
    theta{i}.tau = hps.tau + theta{i}.tau;
    theta{i}.alpha = 1./theta{i}.alpha - 1;
end

% Integrate
y = integ(u, theta, ptheta);

for i = 1:numel(y)
    y{i} = y{i}';
end

end
