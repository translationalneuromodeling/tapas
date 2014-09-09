% Returns the alpha and beta parameters for a Beta density with given mean
% and variance.
%
% See also:
%     betaab2mv

% Kay H. Brodersen, ETH Zurich
% -------------------------------------------------------------------------
function [alpha,beta] = betamv2ab(m,v)
    alpha = m.*(m.*(1-m)./v - 1);
    beta = (1-m).*(m.*(1-m)./v - 1);
    assert(alpha>0 && beta>0, 'distribution parameters out of range');
end
