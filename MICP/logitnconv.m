% Convolves two logit Normal distributions.

% Kay H. Brodersen, ETH Zurich, Switzerland
% http://people.inf.ethz.ch/bkay/
% $Id: logitnconv.m 15572 2012-04-23 12:17:43Z bkay $
% -------------------------------------------------------------------------
function y = logitnconv(res, mu1, sigma1, mu2, sigma2)
    
    % Set support
    x = 0:res:2;
    
    % Individual logit-normal pdfs
    f1 = logitnpdf(x, mu1, sigma1);
    f2 = logitnpdf(x, mu2, sigma2);
    
    % Compute convolution
    y = conv(f1, f2);
    
    % Reduce to [0..2] support
    y = y(1:length(x));
    
    % Normalize (so that all values sum to 1/res)
    y = y / (sum(y) * res);
    
end
