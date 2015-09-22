% PDF of the sum of two independently distributed logit Normal
% distributions.

% Kay H. Brodersen, ETH Zurich, Switzerland
% http://people.inf.ethz.ch/bkay/
% $Id: logitnsumpdf.m 15572 2012-04-23 12:17:43Z bkay $
% -------------------------------------------------------------------------
function y = logitnsumpdf(x, mu1, sigma1, mu2, sigma2)
    
    % Compute convolution
    res = 0.001; % resolution
    c = logitnconv(res, mu1, sigma1, mu2, sigma2);
    
    % Prepare return value
    y = NaN(size(x));
    
    % Fill in return value
    % - values outside support
    y(x<0 | x>2) = 0;
    % - values 
    % - all other values
    idx = int32(x/res+1);
    y(isnan(y)) = c(idx(isnan(y)));
    
end
