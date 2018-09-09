function [op, oy, ollh, olpp, v] = tapas_mpdcm_fmri_sample_hmm(y, u, ...
    theta, ptheta, htheta, oy, op, ollh, olpp)
%% 
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%

T = ptheta.T;
dt = htheta.dt;
nms = htheta.nms;

% Potential energy

np = cell2mat(op);
nx = theta{1}.dim_x;

[nmp, nmt] = size(np);

ms =  ones(nmp, 1);

% Kinetic energies
ok = bsxfun(@times, 1./sqrt(ms), randn(nmp, nmt));
nk = ok;
olaux = -0.5 * sum(bsxfun(@times, ms, nk .* nk), 1);

% Integrate using leap-frogi

% To avoid numerical inaccuracies directly multuply dt inside the gradient.
[dfdp, ny] = gradient(np, y, u, theta, ptheta, 0.5 * dt);
nk = nk - dfdp;

for i = 1:nms
    np = np + bsxfun(@times, dt, bsxfun(@times, nk, ms));
    [dfdp, ny] = gradient(np, y, u, theta, ptheta, dt);
    if i < nms
        nk = nk - dfdp;
    end
end

nk = nk - 0.5 * dfdp;
nk = -nk;

% Metropolis step.

% Parameters
np = mat2cell(np, nmp, ones(1, nmt));

[theta] = tapas_mpdcm_fmri_set_parameters(np, theta, ptheta);

nllh = tapas_mpdcm_fmri_llh(y, u, theta, ptheta, ny);
nlpp = tapas_mpdcm_fmri_lpp(y, u, theta, ptheta, np);

nlaux = -0.5 * sum(bsxfun(@times, ms, nk .* nk), 1);

nllh = sum(nllh, 1);
nlpp = sum(nlpp, 1);


nllh(isnan(nllh)) = -inf;
nlpp(isnan(nlpp)) = -inf;
nlaux(isnan(nlaux)) = -inf;


% Auxiliary

v = nllh .* T + nlpp + nlaux - (ollh .* T + olpp + olaux);
fprintf(1, '%0.5f ', v);
fprintf(1, '\n');

tv = v;
v = rand(size(v)) < exp(v);

ollh(v) = nllh(v);
olpp(v) = nlpp(v);
op(:, v) = np(:, v);
oy(:, v) = ny(:, v);

fprintf(1, '%0.5f ', ollh);
fprintf(1, '\n');

end

function [dp, f] = gradient(op, y, u, theta, ptheta, dt)

dx = 1e-4;

[nmx, nmy] = size(y{1});
% Number of confounds
[nb] = size(ptheta.X0, 2);
[nmb] = nb * nmx;

[nmt] = size(op, 2);
[nmp] = size(op, 1);

vp = mat2cell(op, nmp, ones(1, nmt));

[dfdx, f] = tapas_mpdcm_fmri_gradient_hmm(vp, u, theta, ptheta, ...
    dx, dx);

dp = zeros(nmp, nmt);

ty = y{1}';

mu = full(ptheta.p.theta.mu);
pi = full(diag(ptheta.p.theta.pi));

%
for i = 1:size(op, 2)

    beta = reshape(op(nmp - nmb + 1: nmp, i), nb, nmx);
    f{i} = f{i} + ptheta.X0 * beta;
    % Predicition error
    e = f{i} - ty;
    lambda = op((nmp - nmx - nmb) + 1: (nmp - nmb), i);

    % Gradient of the precision
    dlambda = - 0.5 * sum(e .* e)' .* exp(lambda) + nmy * 0.5;
    
    dp((nmp - nmx - nmb) + 1: (nmp - nmb), i) = dlambda;

    % Multiply by the precisions.
    e = bsxfun(@times, e, exp(lambda)');

    for j = 1:(nmp - nmx - nmb)
        v = -sum(sum(e .* dfdx{i}(:, :, j)));
        dp(j, i) = v; 
    end

    for j = 1:nmx
        o = nmp - nmb + nb * (j - 1);
        dp(o + 1 : o + nb, i) = - ptheta.X0' * e(:, j);
    end

end

%
% Add the gradient of the prior

% Delay the multiplication to this point.
dp(1 : nmp - nmx - nmb, :) = ...
    bsxfun(@times, dp(1 : nmp - nmx - nmb, :), dt .* ptheta.T / dx);

dp(nmp - nmx - nmb + 1 : end, :) = ...
     bsxfun(@times, dp(nmp - nmx - nmb + 1 : end, :), dt .* ptheta.T);
dp = -dp + bsxfun(@times, bsxfun(@times, bsxfun(@minus, op, mu), pi), dt);
%fprintf(1, '%3.5f ', dp(:, end)./dt);
%fprintf(1, '\n');

end
