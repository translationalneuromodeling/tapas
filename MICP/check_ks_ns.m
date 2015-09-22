% Checks input data.

% Kay H. Brodersen, ETH Zurich
% -------------------------------------------------------------------------
function [ks,ns] = check_ks_ns(ks,ns)
    assert(ndims(ks) == 2, 'ks must be a vector or matrix');
    if (size(ks,2)==1 || size(ks,2)==2) && size(ks,1)>2, ks = ks'; ns = ns'; end
    assert(size(ks,1)==1 | size(ks,1)==2, 'ks must have 1 or 2 rows');
    assert(size(ks,2)>1,'ks must have at least two columns');
    assert(size(ks,2)>1,'ns must have at least two columns');
    assert(all(size(ks)==size(ns)),'ks and ns must have same dimensions');
    assert(all(all(ks<=ns)), 'ks cannot be bigger than ns');
    assert(all(all(ks))>=0,'ks must be non-negative');
    assert(all(all(ns))>=0,'ns must be non-negative');
    assert(~all(all(ns==0)),'ns must not be all zero');
end
