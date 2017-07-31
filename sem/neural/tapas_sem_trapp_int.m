function [y] = tapas_sem_trapp_int(u, theta, ptheta, sloppy)
%% Integrate the model of Trappenberg.
%
% Input
%   u       -- Input
%   theta   -- Parameters
%   slopp   -- Optional, if true the input is not checked. Defaults to false.
%       
% Output
%   y       -- Simulated signal.
%   

% aponteeduardo@gmail.com
% copyright (C) 2017
%

n = 3;

n = n + 1;
if nargin < n
    sloppy = 0;
end

if ~sloppy
    tapas_sem_trapp_int_check_input(u, theta, ptheta);
end

y = c_mpdcm_trappenberg_rk4_host(u, theta, ptheta);


end

