function [llh] = tapas_vlinear_hier_llh(data, theta, ptheta)
%% Linear model with variance and identity covariance matrix.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

y = data.y;
u = data.u;

theta = theta.y;

[np, nc] = size(y);

llh = zeros(np, nc);

ln2pi = log(2 * pi);

for j = 1:nc
    % Precision
    mu = theta{j}.mu;
    pe = theta{j}.pe;
    lpe = log(pe);

   for i = 1:np
        r = y{i, j} - mu;
        n = numel(r);        
        llh(i, j) = - 0.5 * n * ln2pi + 0.5 * n * lpe - 0.5 * pe * sum(r .* r);
    end
end


end

