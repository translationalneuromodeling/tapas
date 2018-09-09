function [nhtheta] = tapas_ti_init_htheta(ptheta, htheta, pars)
% Init the parameters of the sampler

nhtheta = htheta;

% It is better not to adapt certain parameteres
if ~isfield(htheta, 'mixed')
    nhtheta.mixed = ones(size(ptheta.jm, 1), 1);
end

nhtheta.nmixed = abs(nhtheta.mixed - 1);
nhtheta.knmixed = chol(htheta.pk)' * ptheta.jm;

nhtheta.ok = init_kernel(ptheta, nhtheta, pars);

end

function [nk] = init_kernel(ptheta, htheta, pars)
%% Initilize the kernel or covariance matrix of the proposal distribution.
%
% See Exploring an adaptative Metropolis Algorithm
% 

T = pars.T;
s0 = 0.05;

np = size(htheta.pk, 1); 

njm = tapas_ti_zeromat(ptheta.jm);

c = njm' * htheta.pk * njm;
c = chol(c);

nk = cell(numel(T), 1);
nk(:) = {c};

k =  s0 * chol(htheta.pk)' * ptheta.jm;
tk = cell(numel(T), 1);
tk(:) = {k};
nk = struct('S', nk, 's', s0, 'k', tk);

end


