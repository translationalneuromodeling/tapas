function [lpp] = tapas_ti_lpp(y, x, u, theta, ptheta)
%% General prior probability.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

flpp = ptheta.method_lpp;

ns = size(theta, 1);
nc = size(theta, 2);

lpp = zeros(1, nc);

for i = 1:nc
    tlpp = zeros(ns, 1);
    for j = 1:ns
        tlpp(j) = flpp(y{j, i}, x{j, i}, u{j, i}, theta{j, i}, ptheta);
    end
    lpp(i) = sum(tlpp);
end

end

