% PDF of the average of two independent random variables which are
% distributed according to logit Normal distributions.

% Kay H. Brodersen, ETH Zurich, Switzerland
% http://people.inf.ethz.ch/bkay/
% $Id: logitnavgpdf.m 15572 2012-04-23 12:17:43Z bkay $
% -------------------------------------------------------------------------
function y = logitnavgpdf(x, mu1, sigma1, mu2, sigma2)
    
    assert(sigma1>0 && sigma2>0,'sigma must be positive');
    assert(isscalar(mu1) && isscalar(mu2),'mu must be a scalar');
    assert(isscalar(sigma1) && isscalar(sigma2),'sigma must be a scalar');
    
    y = logitnsumpdf(2*x, mu1, sigma1, mu2, sigma2);
    y = y*2;
    
end
