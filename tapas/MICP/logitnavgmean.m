% Expectation of the average of two logit-normal densities.

% Kay H. Brodersen, ETH Zurich, Switzerland
% http://people.inf.ethz.ch/bkay/
% $Id: logitnavgmean.m 16157 2012-05-28 09:24:16Z bkay $
% -------------------------------------------------------------------------
function mu_phi = logitnavgmean(mu1,sigma1,mu2,sigma2)
    
    assert((isscalar(mu1) && isscalar(sigma1)) || (isvector(mu1) && isvector(sigma1) && all(size(mu1)==size(sigma1))));
    assert(all(size(mu1)==size(mu2)));
    assert(all(size(sigma1)==size(sigma2)));
    
    res = 0.001;
    x = 0:res:2;
    
    mu_phi = NaN(size(mu1));
    for i=1:length(mu1)
        c = logitnconv(res,mu1(i),sigma1(i),mu2(i),sigma2(i));
        mu_phi(i) = sum(x.*c/2) * res;
    end
    
end
