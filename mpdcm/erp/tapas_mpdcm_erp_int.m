function [y] = tapas_mpdcm_erp_int(u, theta, ptheta, sloppy)
%% 
%
% Input
%
% Output
%

% aponteeduardo@gmail.com
% copyright (C) 2016
%


if nargin < 4
    sloppy = 0;
end

if ~sloppy
    assert(tapas_mpdcm_num_devices() > 0, 'mpdcm:fmri:int:no_gpu', ...
        'No GPU device available.');
    tapas_mpdcm_erp_int_check_input(u, theta, ptheta);
end

if isfield(ptheta, 'integ')
    switch ptheta.integ
    case 'euler'
        integ = @c_mpdcm_erp_euler;
    case 'rk4'
        integ = @c_mpdcm_erp_rk4;
    otherwise
        error('mpdcm:fmri:int:input', ... 
            'Unknown method for ptheta.int');
end
else
    error('mpdcm:fmri:int:input', 'Integrator must be specified.');
end

% Integrate

y = integ(u, theta, ptheta);

end % tapas_mpdcm_erp_int 

