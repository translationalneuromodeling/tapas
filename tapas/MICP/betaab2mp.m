% Returns the alpha and beta parameters for a Beta density with given mean
% and precision.

% Kay H. Brodersen, ETH Zurich
% -------------------------------------------------------------------------
function [m,p] = betaab2mp(alpha,beta)
    [m,v] = betaab2mv(alpha,beta);
    p = 1/v;
end
