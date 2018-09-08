function [ptheta] = tapas_sem_prepare_gaussian_ptheta(ptheta)
%% Precomputes things that might be needed afterwards.
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%

np = ptheta.ndims;
dtheta = size(ptheta.jm, 1)/np;

LN2PI = log(2 * pi);

njm = tapas_zeromat(ptheta.jm);

% Compute the values that are beta distributed
bdist = zeros(np, 1);
bdist(ptheta.bdist) = 1;
bdist = kron(ones(dtheta, 1), bdist);
ptheta.bdist = find(bdist);      

% Compute the values that are beta distributed
njm = bsxfun(@times, njm, 1 - bdist);

% Compute the number of parameters 
njm = njm(:, any(njm, 1));
np = sum(njm(:));

kjm = njm * njm';

ptheta.pm = kron(ones(dtheta, 1), ptheta.pm);
ptheta.mu = kron(ones(dtheta, 1), ptheta.mu);
ptheta.p0 = kron(ones(dtheta, 1), ptheta.p0);

ptheta.pconst = sum(log(njm' * ptheta.pm)) - 0.5 * np * LN2PI;

% Better to keep a sparse representation
ptheta.kjm = sparse(kjm * diag(ptheta.pm) * kjm);

% Check if it's diagonal and save time
if all((ptheta.kjm - diag(diag(ptheta.kjm))) == 0)
    ptheta.dkjm = diag(ptheta.kjm);
end

end

