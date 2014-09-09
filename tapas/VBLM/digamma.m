function dg=digamma(x,h)
% DIGAMMA - numerical approximation to digamma function 0.
% 
% DG=DIGAMMA(X)
% DG=DIGAMMA(X,H)
% 
% evaluates gamma'(x)/gamma(x) using a numerical approximation 
% to the derivative. Step size in the numerical approximation is H
% (default 0.00001)
%
%        (C) T. Behrens 2002 

if(nargin==1);h=0.00001;end
gamdash=(gamma(x-2*h)-8*gamma(x-h)+8*gamma(x+h)-gamma(x+2*h))/12/h;
dg=gamdash./gamma(x);
