function [llh] = tapas_sem_prosa_cllh(t, a, u, theta, method, sloppy)
%% Computes the likelihood function of a trial using the prosa model. It
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
% Prosaccades happen when:
%    a > s > p
%    s > a > p
%
% Antisaccades happen when:
%    a > p > s
%    p > a > s
%    p > s > a
%    s > p > a
%
% aponteeduardo@gmail.com
% copyright (C) 2015
%

n = 6;

if nargin < n
    sloppy = 0;
end

if ~sloppy
    tapas_sem_prosa_llh_check_input(t, a, u, theta);
end

llh = method(t, a, u, theta);

end
