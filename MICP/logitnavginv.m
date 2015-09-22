% Inverse CDF of the average of two independent random variables which are
% distributed according to logit-normal distributions.
%
% If the optimization fails, function returns NaN.

% Kay H. Brodersen, ETH Zurich, Switzerland
% http://people.inf.ethz.ch/bkay/
% $Id: logitnavginv.m 16157 2012-05-28 09:24:16Z bkay $
% -------------------------------------------------------------------------
function x = logitnavginv(y,mu1,sigma1,mu2,sigma2)
    
    assert(isscalar(y));
    assert((isscalar(mu1) && isscalar(sigma1)) || (isvector(mu1) && isvector(sigma1) && all(size(mu1)==size(sigma1))));
    assert(all(size(mu1)==size(mu2)));
    assert(all(size(sigma1)==size(sigma2)));
    
    x = NaN(size(mu1));
    for i=1:length(mu1)
        try
            x(i) = fzero(@(z) logitnavgcdf(z,mu1(i),sigma1(i),mu2(i),sigma2(i))-y, 0.5);
        catch
            disp(['Error occurred in LOGITNAVGINV']);
            x(i) = NaN;
        end
    end
    
end
