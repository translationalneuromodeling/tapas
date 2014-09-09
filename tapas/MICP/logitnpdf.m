% Probability density function of the logit-normal distribution.

% Kay H. Brodersen, ETH Zurich, Switzerland
% $Id: logitnpdf.m 14789 2012-03-05 09:06:58Z bkay $
% -------------------------------------------------------------------------
function y = logitnpdf(x,mu,sigma)
    assert(sigma>0,'sigma must be positive');
    assert(isscalar(mu),'mu must be a scalar');
    assert(isscalar(sigma),'sigma must be a scalar');
    
    y = 1/(sigma*sqrt(2*pi)) * exp(-((logit(x)-mu).^2./(2*sigma^2))) ./ (x.*(1-x));
    y(isnan(y)) = 0;
end
