function [llh] = tapas_sem_multiv_llh(data, theta, ptheta)
%% Compute likelihood of the eye movement model in a hierarchical format. 
%
%   Input
%
%   data        -- Stucture array of dimension Nx1 with fields y and u
%   theta       -- Structure array of dimension 1 X 1 
%   ptheta      -- Priors
%
%   Ouput
%   
%   llh         -- Matrix of dimensions NxM where N is the number fo subjects
%                   and M the number of chains.
%

% aponteeduardo@gmail.com
% copyright (C) 2016
%


ns = size(theta.y, 1);
nc = size(theta.y, 2);

atheta = cell(ns, nc);

for i = 1:ns
    for j = 1:nc
        atheta{i, j} = ptheta.model.p0 + ptheta.model.jm * theta.y{i, j};
    end
end

llh =  ptheta.model.llh(data, atheta);

end

