function [llh] = tapas_vlinear_llh(data, theta, ptheta)
%%  
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

y = data.y;
u = data.u;

theta = theta.y;

[~, nc] = size(y);

llh = zeros(1, nc);
ln2pi = log(2 * pi);

for i = 1:nc
    pe = y{i}.pe;
    lpe = log(pe);
    tllh = theta{i}.k * log(theta{i}.t) - gammaln(theta{i}.k) - ...
        pe * theta{i}.t + (theta{i}.k - 1) * log(pe);

    r = theta{i}.mu - y{i}.mu;
    n = numel(r);        
    llh(1, i) = tllh - 0.5 * n * ln2pi + 0.5 * n * lpe ...
        - 0.5 * pe * sum(r .* r);
end

end

