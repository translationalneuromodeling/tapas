function [llh] = tapas_dlinear_hier_llh(data, theta, ptheta)
%% Likelihood of the nodes of a model with diagonal precision matrix.
% Note that this is the likelihood of several parameters theta 1 to n with mean
% mu and precision matrix diag(pe).
%

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
    % Mean
    mu = theta{j}.mu;
    % Precision
    pe = theta{j}.pe;
    lpe = log(pe);

   for i = 1:np
        r = y{i, j} - mu;
        llh(i, j) = sum(- 0.5 * ln2pi + 0.5 * lpe - 0.5 * pe .* r .* r);
    end
end


end

