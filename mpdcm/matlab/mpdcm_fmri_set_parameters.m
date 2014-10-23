function [theta] = mpdcm_fmri_set_parameters(p, theta, ptheta)
%% Sets the parameters introduced in vectorial form
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%


nt = numel(theta);
nl = numel(ptheta.Q);

for i = 1:nt

    nr = size(ptheta.a, 1);   
    tp = p{i};

    oi = 0;
    ni = sum(logical(ptheta.a(:)));

    theta{i}.A(logical(ptheta.a)) = indexing(tp, oi, ni);
    
    for j = 1:size(ptheta.b, 3)
        t = logical(ptheta.b(:, :, j));
        oi = ni;
        ni = oi + sum(t(:));
        theta{i}.B{j}(t) = indexing(tp, oi, ni);
    end

    oi = ni;
    ni = oi + sum(logical(ptheta.c));
    theta{i}.C(logical(ptheta.c)) = indexing(tp, oi, ni);

    oi = ni;
    ni = oi + nr;
    theta{i}.K = indexing(tp, oi, ni);

    oi = ni;
    ni = oi + nr;
    theta{i}.tau = indexing(tp, oi, ni);

    oi = ni;
    ni = oi + 1;
    theta{i}.epsilon = indexing(tp, oi, ni);

    oi = ni;
    ni = oi + nl;
    theta{i}.lambda = indexing(tp, oi, ni);

end

end

function [na] = indexing(a, li, hi )

if li == hi
    na = [];
    return
end

na = a(li+1:hi);

end
