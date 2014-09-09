% Cumulative density function of the logit-normal distribution.

% Kay H. Brodersen, ETH Zurich, Switzerland
% $Id: logitncdf.m 14789 2012-03-05 09:06:58Z bkay $
% -------------------------------------------------------------------------
function p = logitncdf(x,mu,sigma)
    if isnan(mu) || isnan(sigma), p=NaN; return; end
    assert(isscalar(mu),'mu must be a scalar');
    assert(isscalar(sigma),'sigma must be a scalar');
    assert(sigma>0,'sigma must be positive');
    
    p = 1/2.*(1+erf((logit(x)-mu)./(sqrt(2)*sigma)));
    p(x>=1) = 1;
    p(x<=0) = 0;
end
