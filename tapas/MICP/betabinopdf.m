% Beta-binomial probability density function.
% 
% Usage:
%     betabinopdf(x,n,alpha,beta)

% Kay H. Brodersen, ETH Zurich
%
% Uses code from:
% Trujillo-Ortiz, A., R. Hernandez-Walls, F.A. Trujillo-Perez and 
%   N. Castro-Castro (2009). bbinopdf:Beta-binomial probability 
%   densiy function. A MATLAB file. [WWW document]. URL 
%   http://www.mathworks.com/matlabcentral/fileexchange/25454
% -------------------------------------------------------------------------
function y = betabinopdf(x,n,a,b)

    y = exp(gammaln(n + 1)-gammaln(x + 1)-gammaln(n - x + 1)) .* beta((a + x),(b + n - x)) ./ beta(a,b);

end
