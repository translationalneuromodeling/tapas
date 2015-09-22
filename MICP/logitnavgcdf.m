% CDF of the average of two independent random variables which are
% distributed according to logit-normal distributions.

% Kay H. Brodersen, ETH Zurich, Switzerland
% http://people.inf.ethz.ch/bkay/
% $Id: logitnavgcdf.m 15572 2012-04-23 12:17:43Z bkay $
% -------------------------------------------------------------------------
function y = logitnavgcdf(x,mu1,sigma1,mu2,sigma2)
    
    y = logitnsumcdf(2*x,mu1,sigma1,mu2,sigma2);
    
end
