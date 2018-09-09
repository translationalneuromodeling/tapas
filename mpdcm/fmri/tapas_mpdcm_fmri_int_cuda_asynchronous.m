function [container] = tapas_mpdcm_fmri_int_cuda_asynchronous(u, theta, ptheta, sloppy)
%% Integrates the system of differential equations specified by the input.
%
% Input:
% u         -- Cell array of experimental input
% theta     -- Cell array of model parameteres
% ptheta    -- Structure. Model priors or constants
% sloppy    -- Scalar. If true don't check the input. 
%
% Ouput:
% cotainer  -- A pointer to a container object.
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
    assert(tapas_mpdcm_num_devices() > 0, 'mpdcm:fmri:int:no_gpu', ...
        'No GPU device available.');
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);  
end

if isfield(ptheta, 'integ')
    switch ptheta.integ
    case 'adfadfa8esdfas'
        if theta{1}.fD
            integ = @(u, theta, ptheta) c_mpdcm_fmri_euler_nonlinear(u, ...
                theta, ptheta);
        else 
            integ = @(u, theta, ptheta) c_mpdcm_fmri_euler_bilinear(u, ...
                theta, ptheta);
        end
    case 'rk4'
        if theta{1}.fD
            integ = @(u, theta, ptheta) ...
                c_mpdcm_fmri_std_rk4_nonlinear_asynchronous(u, ...
                theta, ptheta);
        else 
            integ = @(u, theta, ptheta) ...
                c_mpdcm_fmri_std_rk4_bilinear_asynchronous(u, ...
                theta, ptheta);
        end
    otherwise
        error('tapas:mpdcm:fmri:int:input', 'Unknon integrator.');
    end
else
        error('mpdcm:fmri:int:input', 'Integrator must be specified.');
end

hps = tapas_mpdcm_fmri_get_hempars();
for i = 1:numel(theta)
    [k1, k2, k3] = tapas_mpdcm_fmri_k(theta{i});
    theta{i}.k1 = k1;
    theta{i}.k2 = k2;
    theta{i}.k3 = k3;
    % Change the parametrization
    theta{i}.K = hps.K * exp(theta{i}.K);
    theta{i}.tau = hps.tau + theta{i}.tau;
end

% Integrate
container = integ(u, theta', ptheta);

end
