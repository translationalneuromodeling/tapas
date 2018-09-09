function [ntheta] = tapas_mpdcm_erp_transform_theta_host(theta, ptheta)
%% Tranforms the data to different representations. 
%
% Input
%   theta       Array of thetas
%
% Output
%   ntheta      Array of mpdcm compatible data structures
%

% aponteeduardo@gmail.com
% copyright (C) 2016
%


nt = numel(theta);
ntheta = cell(size(theta));

for i = 1:nt
    ttheta = rescale_theta(theta{i});
    ntheta{i} = transform_theta(ttheta);
end

end % tapas_mpdcm_erp_transform_theta 

function [ntheta] = transform_theta(theta)

    nl = size(theta.T, 1);
    I = eye(nl);
    ntheta = struct();

	ntheta.dim_x = nl;
	ntheta.dim_u = 1;

	% Note: The weighting can be precomputed for gamma 2 and gamma 4 
	ntheta.gamma1 = full(theta.G(:, 1) .* theta.He ./ theta.Te);
	ntheta.gamma2 = full(theta.G(:, 2) .* theta.He ./ theta.Te);
	ntheta.gamma3 = full(theta.G(:, 3) .* theta.He ./ theta.Te);
	ntheta.gamma4 = full(theta.G(:, 4) .* theta.Hi ./ theta.Ti);

	ntheta.tau_e2 = full(2./ theta.Te);
	ntheta.tau_i2 = full(2./ theta.Ti);

	ntheta.tau_es2 = full(1./theta.Te.^2);
	ntheta.tau_is2 = full(1./theta.Ti.^2);

	ntheta.r1 = full(theta.S(1));
	ntheta.r2 = full(theta.S(2));

	ntheta.er1r2 = full(1 / ( 1 + exp(theta.S(1) * theta.S(2))));

	ntheta.Au = full(theta.He ./ theta.Te .* theta.C);

	% Forward and lateral
	ntheta.A13 = bsxfun(@times, full(theta.He ./ theta.Te), ...
		full(theta.A{1} + theta.A{3}));
	% Back and lateral
	ntheta.A23 = bsxfun(@times, full(theta.He ./ theta.Te), ...
		full(theta.A{2} + theta.A{3})); 

    ntheta.C = full(theta.C);

end % transform_theta

function [ntheta] = rescale_theta(theta, ptheta)
% Rescales parameters (matches spm)

% Number of regions
nr = size(theta.A{1}, 1);

E = [1 1/2 1/8] * 32;         % extrinsic rates (forward, backward, lateral)
G = [1 4/5 1/4 1/4] * 128;  % intrinsic rates (g1 g2 g3 g4)
D = [2 16];                 % propogation delays (intrinsic, extrinsic)
H = [4 32];                 % receptor densities (excitatory, inhibitory)
T = [8 16];                 % synaptic constants (excitatory, inhibitory)
R = [2 1]/3;                % parameters of static nonlinearity

try, E = ptheta.pF.E; end
try, G = ptheta.pF.G; end
try, D = ptheta.pF.D; end
try, H = ptheta.pF.H; end
try, T = ptheta.pF.T; end
try, R = ptheta.pF.R; end

ntheta = theta;

ntheta.A{1} = E(1) * exp(theta.A{1});
ntheta.A{2} = E(2) * exp(theta.A{2});
ntheta.A{3} = E(3) * exp(theta.A{3});

try
    G = G .* exp(theta.G);
end

ntheta.G = ones(nr, 1) * G;

ntheta.He = ones(nr, 1) * H(1) * exp(theta.H(:, 1));
ntheta.Hi = ones(nr, 1) * H(2) * exp(theta.H(:, 2));

ntheta.Te = T(1)/1000 * exp(theta.T(:, 1));
ntheta.Ti = T(2)/1000 * exp(theta.T(:, 2));

ntheta.S = R .* exp(theta.S);

ntheta.C = exp(theta.C);

end
