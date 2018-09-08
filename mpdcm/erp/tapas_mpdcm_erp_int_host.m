function [y] = tapas_mpdcm_erp_int_host(u, theta, ptheta, sloppy)
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
    tapas_mpdcm_erp_int_check_input_host(u, theta, ptheta);
end

if isfield(ptheta, 'integ')
    switch ptheta.integ
    case 'rk4'
        integ = @c_mpdcm_erp_rk4_host;
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

