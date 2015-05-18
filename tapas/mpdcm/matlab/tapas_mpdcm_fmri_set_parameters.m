function [theta] = tapas_mpdcm_fmri_set_parameters(p, theta, ptheta)
% Sets the parameters introduced in vectorial form.
%
% Input:
% p     -- Cell array of parameters in matrix form
% theta     -- Cell array of parameters in structure form
% ptheta    -- Hyperparameters
%
% Ouput:
% thate     -- Cell array of parameters in structure form
%

% aponteeduardo@gmail.com
%
% Author: Eduardo Aponte, TNU, UZH & ETHZ - 2015
% Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
%
% Licensed under GNU General Public License 3.0 or later.
% Some rights reserved. See COPYING, AUTHORS.
%
% Revision log:
%
%


nt = numel(theta);
nl = numel(ptheta.Q);

for i = 1:nt

    nr = size(ptheta.a, 1);   
    tp = p{i};

    oi = 0;
    ni = sum(logical(ptheta.a(:)));

    theta{i}.A(logical(ptheta.a)) = indexing(tp, oi, ni);

    oi = ni;
    ni = oi + sum(logical(ptheta.b(:)));
    theta{i}.B(ptheta.b) = indexing(tp, oi, ni);
        
    oi = ni;
    ni = oi + sum(logical(ptheta.c(:)));
    theta{i}.C(logical(ptheta.c)) = indexing(tp, oi, ni);

    oi = ni;
    ni = oi + sum(logical(ptheta.d(:)));
    theta{i}.D(logical(ptheta.d)) = indexing(tp, oi, ni);

    oi = ni;
    ni = oi + nr;
    theta{i}.K = indexing(tp, oi, ni);

    oi = ni;
    ni = oi + nr;
    theta{i}.tau(:) = indexing(tp, oi, ni);

    oi = ni;
    ni = oi + 1;
    theta{i}.epsilon(:) = indexing(tp, oi, ni);

    oi = ni;
    ni = oi + nl;
    theta{i}.lambda = indexing(tp, oi, ni);

end

end

function [na] = indexing(a, li, hi )

% Empty array
if li == hi
    na = [];
    return
end

na = a(li+1:hi);

end
