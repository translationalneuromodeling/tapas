% Returns the alpha and beta parameters for a Beta density with given mean
% and variance.

% Kay H. Brodersen, ETH Zurich
% -------------------------------------------------------------------------
function [m,v] = betaab2mv(alpha,beta)
    assert(alpha>0 && beta>0, 'distribution parameters out of range');
    m = alpha./(alpha+beta);
    v = alpha.*beta./((alpha+beta).^2 .* (alpha+beta+1));
end
