function [pp, pa, ap, aa] = tapas_sem_seri_gen_lh(t, theta, ptheta)
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

[PROSACCADE ANTISACCADE] = tapas_sem_action_codes();

DIM_THETA = tapas_sem_seri_ndims();

DEL = 17;

nt = numel(t);

tt = zeros(nt, 1);
a = zeros(nt, 1);

method = ptheta.method;

%

tt(:) = PROSACCADE;
a(:) = PROSACCADE;


pp = tapas_sem_seri_cllh(t, a, tt, theta, method);

%

tt(:) = ANTISACCADE;
a(:) = PROSACCADE;

ap = tapas_sem_seri_cllh(t, a, tt, theta, method);

%

tt(:) = PROSACCADE;
a(:) = ANTISACCADE;

pa = tapas_sem_seri_cllh(t, a, tt, theta, method);

%

tt(:) = ANTISACCADE;
a(:) = ANTISACCADE;

aa = tapas_sem_seri_cllh(t, a, tt, theta, method);

end
