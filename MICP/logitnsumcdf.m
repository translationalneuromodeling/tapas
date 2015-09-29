% CDF of the sum of two independent random variables which are distributed
% according to logit-normal distributions.
% 
% Note: When computing multiple values at once, 'betaconv' contains a more
% efficient implementation.

% Kay H. Brodersen, ETH Zurich, Switzerland
% http://people.inf.ethz.ch/bkay/
% $Id: logitnsumcdf.m 15572 2012-04-23 12:17:43Z bkay $
% -------------------------------------------------------------------------
function y = logitnsumcdf(x,mu1,sigma1,mu2,sigma2)
    
    if ~(ndims(x)==2 && (size(x,1)==1 || size(x,2)==1))
        error('only implemented for onedimensional input');
    end
    
    % Compute the PDF first (since we want the entire pdf rather than just
    % one value from it, using betaconv is computationally more efficient
    % than using betasumpdf)
    res = 0.001;
    c = logitnconv(res,mu1,sigma1,mu2,sigma2);
    
    % Sum the PDF up to point x
    for i=1:length(x)
        idx = round(x(i)/res);
        if idx < 1
            y(i) = 0;
        elseif idx > length(c)
            y(i) = 1;
        else
            y(i) = trapz(c(1:idx)) * res;
        end
    end
    
end
