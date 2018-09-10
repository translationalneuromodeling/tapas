function [nx] = tapas_ti_gen_state(y, x, u, theta, ptheta)
%% General method to generate state space.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

gen_state = ptheta.method_state;

ns = size(theta, 1);
nc = size(theta, 2);

nx = cell(ns, nc);

for i = 1:ns
    for j = 1:nc
        nx{i, j} = gen_state(y{i, j}, [], u{i, j}, theta{i, j}, ptheta);
    end
end

end

