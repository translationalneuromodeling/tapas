function [lpp] = tapas_sem_prosa_lpp(theta, ptheta)
%% Log prior probability of the parameters of the prosa model. 
%
% Input
%   pv -- Parameters in numerical form.
%   theta -- Parameters of the model
%   ptheta -- Priors of parameters
%
% Output
%   lpp -- Log prior probability
%

% aponteeduardo@gmail.com
% copyright (C) 2015
%

nconst = ptheta.pconst;

lpp = zeros(1, numel(theta));
for i = 1:numel(theta)
    if isfield(ptheta, 'dkjm')
        err = sum((ptheta.mu - theta{i}) .* ptheta.dkjm .* ...
        (ptheta.mu - theta{i}));
    else 
        err = (ptheta.mu - theta{i})' * ptheta.kjm * (ptheta.mu - theta{i});
    end

    p0 = atan(theta{i}(ptheta.bdist))/pi + 0.5;
    alpha = ptheta.mu(ptheta.bdist);
    beta = ptheta.pm(ptheta.bdist);
    lpp(i) = nconst - 0.5 * err; + ...
        sum(betaln(alpha, beta) + ...
            (alpha - 1) .* log(p0) + (beta - 1) .* log(1 - p0));
end

end
