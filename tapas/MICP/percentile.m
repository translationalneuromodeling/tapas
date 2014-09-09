function [y] = percentile(x,p)
% y = percentile(x,p)
%   p is a number between 0 and 100 (the percentile value)
%   x is the data vector (it will reshape any matrix into a single column)

if (p>100), p=100; end
if (p<0),   p=0;   end
n = prod(size(x));
xx = sort(reshape(x,n,1));
y = xx(1 + round((n-1)*p/100));
