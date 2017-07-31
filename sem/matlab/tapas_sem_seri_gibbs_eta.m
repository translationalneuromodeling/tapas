function [otheta] = tapas_sem_seri_gibbs_eta(y, u, otheta, ptheta, T)
%% Uses a Gibbs step to sample the eta parameters. 
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2017
%

nt = numel(T);
alpha = ptheta.mu(11);
beta = ptheta.pm(11);
for i = 1:nt
    d0p = exp(otheta{i}(9));
    d0a = exp(otheta{i}(20));
    vals = T(i) * [d0p > y.t(u.tt == 0), d0p <= y.t(u.tt == 0);
        d0a > y.t(u.tt == 1), d0a <= y.t(u.tt == 1)];
    vals = sum(vals, 1) + [alpha, beta];
    p = betarnd(vals(1), vals(2));
    otheta{i}([11, 22]) = log(p / (1-p));
end

end

