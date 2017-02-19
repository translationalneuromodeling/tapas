function [pp, pa, ap, aa] = tapas_sem_prosa_gen_lh(x, theta, ptheta)
%% Generate the likelihood function. 
%
% Input
%   x       Time interval to plot the likelihood function.
%   theta   Parameters to use
%   ptheta  Priors
%
% Output
%   pp      Prosaccade trials, prosaccade action
%   pa      Prosaccade trial, antisaccade action
%   ap      Antisaccade trial, prosaccade action
%   aa      Antisaccade trial, antisaccade action
%

% aponteeduardo@gmail.com
% copyright (C) 2015
%

[PROSACCADE, ANTISACCADE] = tapas_sem_action_codes();

DIM_THETA = tapas_sem_prosa_ndims();

nt = numel(x);

u = struct('tt', zeros(nt, 1));
y = struct('t', x, 'a', zeros(nt, 1));

%
method = ptheta.method;

u.tt(:) = PROSACCADE;
y.a(:) = PROSACCADE;

pp = exp(method(y.t, y.a, u.tt, theta));

%

u.tt(:) = ANTISACCADE;
y.a(:) = PROSACCADE;

ap = exp(method(y.t, y.a, u.tt, theta));

%

u.tt(:) = PROSACCADE;
y.a(:) = ANTISACCADE;

pa = exp(method(y.t, y.a, u.tt, theta));

%

u.tt(:) = ANTISACCADE;
y.a(:) = ANTISACCADE;

aa = exp(method(y.t, y.a, u.tt, theta));

end
