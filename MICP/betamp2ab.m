% Returns the alpha and beta parameters for a Beta density with given mean
% and precision.

% Kay H. Brodersen, ETH Zurich
% -------------------------------------------------------------------------
function [alpha,beta] = betamp2ab(m,p)
    v = 1/p;
    [alpha,beta] = betamv2ab(m,v);
end
