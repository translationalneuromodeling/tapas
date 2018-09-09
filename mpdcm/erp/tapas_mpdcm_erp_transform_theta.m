function [ntheta] = tapas_mpdcm_erp_transform_theta(theta, ptheta)
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
    ntheta = cell(1, nl);

    for i = 1:nl
        ntheta{i}.dim_x = nl;
        ntheta{i}.dim_u = 1;


	% Note: The weighting can be precomputed for gamma 2 and gamma 4 
        ntheta{i}.gamma1 = full(theta.G(i, 1)) * theta.He(i) / theta.Te(i));
        ntheta{i}.gamma2 = full(theta.G(i, 2) * theta.He(i) / theta.Te(i));
        ntheta{i}.gamma3 = full(theta.G(i, 3)) * theta.He(i) / theta.Te(i));
        ntheta{i}.gamma4 = full(theta.G(i, 4) * theta.Hi(i) / theta.Ti(i));

        ntheta{i}.tau_e2 = full(2./ theta.Te(i));
        ntheta{i}.tau_i2 = full(2./ theta.Ti(i));

        ntheta{i}.tau_es2 = full(1./theta.Te(i).^2);
        ntheta{i}.tau_is2 = full(1./theta.Ti(i).^2);
    
        ntheta{i}.r1 = full(theta.S(1));
        ntheta{i}.r2 = full(theta.S(2));

        ntheta{i}.er1r2 = full(1 / ( 1 + exp(theta.S(1) * theta.S(2))));
    
        ntheta{i}.Au = full(theta.He(i) * theta.C(i));
    
        % Forward and lateral
        ntheta{i}.A13 = full((theta.He(i)/theta.Te(i)) * ...
            (theta.A{1}(i, :) + theta.A{3}(i, :)))';
        % Back and lateral
        ntheta{i}.A23 = full((theta.He(i) / theta.Te(i)) * ...
            (theta.A{2}(i, :) + theta.A{3}(i, :)))'; 

    end

    ntheta = struct('dim_x', nl, 'dim_u', 1, 'columns', {ntheta}, ...
        'theta', theta);

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
