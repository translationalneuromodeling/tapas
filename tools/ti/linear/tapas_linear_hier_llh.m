function [llh] = tapas_linear_hier_llh(data, theta, ptheta)
%%  
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

% Precision
pe = ptheta.pe;
lpe = log(ptheta.pe);

for j = 1:nc;
    for i = 1:np
        r = y{i, j} - theta{j};
        n = numel(r);        
        llh(i, j) = - 0.5 * n * ln2pi + 0.5 * n * lpe - 0.5 * pe * sum(r .* r);
    end
end


end

