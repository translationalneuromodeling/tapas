function [llh] = tapas_sem_seri_cllh(t, a, u, theta, method, sloppy)
%% Computes the likelihood function of a trial using the seri model. It
% relies on a c implementation.
%
% Input
%   t       Reaction time. Should be a positive number
%   a       Action, should be 0 (prosaccade) or 1 (antisaccade).
%   u       Experimental conditions.
%   theta   Model parameters.
%   sloppy  If sloppy, parameters are not check before hand. Defaults to 
%           False
%
% Output
%   llh -- Log likelihood of the trial
%

% aponteeduardo@gmail.com
% copyright (C) 2015
%

n = 6;

if nargin < n
    sloppy = 0;
end

if ~sloppy
    tapas_sem_seri_llh_check_input(t, a, u, theta);
end

llh = method(t, a, u, theta);

end
