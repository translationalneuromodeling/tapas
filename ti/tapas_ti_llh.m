function [llh] = tapas_ti_llh(y, x, u, theta, ptheta)
%% General method to iterate across chains and subjects 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

fllh = ptheta.method_llh;

ns = size(theta, 1);
nc = size(theta, 2);

llh = zeros(1, nc);

for i = 1:nc
    tllh = zeros(ns, 1);
    for j = 1:ns
        tllh(j) = fllh(y{j, i}, x{j, i}, u{j, i}, theta{j, i}, ptheta);
    end
    llh(i) = sum(tllh);
end

%if any(llh < 1000)
%    keyboard
%end

end

