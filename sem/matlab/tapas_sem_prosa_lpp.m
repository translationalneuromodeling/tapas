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

    alpha = ptheta.mu(ptheta.bdist); 
    beta = ptheta.pm(ptheta.bdist);
    lcosh = log(cosh(theta{i}(ptheta.bdist)/2)); 
    lpp(i) = nconst - 0.5 * err + ...
       ...sum(betaln(alpha, beta) + ...
        sum((alpha - 1) .* (theta{i}(ptheta.bdist)/2 - log(2) - ...
            lcosh) + ...
        (beta - 1) .* (-theta{i}(ptheta.bdist)/2 - log(2) - ...
            lcosh)) - ...
        2 * sum(log(2) + lcosh);
end

end
