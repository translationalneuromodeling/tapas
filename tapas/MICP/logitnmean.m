% Expectation (mean) of a logitnormal distribution.
% 
% Usage:
%     e = logitnmean(mu,sigma)
% 
% Arguments:
%     mu: location parameter (mean of the logit of x)
%     sigma: scale parameter (standard deviation of the logit of x)
%
% Both arguments may be vectors, in which case the result is also a vector.
% Function uses numerical integration. Poor approximation when sigma large
% (resulting in most of the probability mass being close to 0 and close to
% 1).

% Kay H. Brodersen, ETH Zurich
% $Id: logitnmean.m 18773 2013-01-22 17:10:37Z bkay $
% -------------------------------------------------------------------------
function e = logitnmean(mu,sigma)
    
    assert((isscalar(mu) && isscalar(sigma)) || (isvector(mu) && isvector(sigma) && all(size(mu)==size(sigma))));
    if any(sigma>1), disp('logitnmean: WARNING: may want to use a finer grid for sigma > 1'); end
    grid = 0:0.0001:1;
    for i=1:length(mu)
        if (isnan(mu(i)) || isnan(sigma(i)) || sigma(i) <= 0)
            e(i) = NaN;
        else
            values = grid.*logitnpdf(grid,mu(i),sigma(i));
            e(i) = trapz(grid,values);
        end
    end
    
end
