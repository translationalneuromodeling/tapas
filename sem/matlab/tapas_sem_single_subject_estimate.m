function [ps, fe] = ...
    tapas_sem_single_subject_estimate(data, ptheta, htheta, pars)
%% Estimates the posterior probability of the parameters using MCMC combined
% with path sampling.
%
% Input
%   y       Saccade data.
%   u       Experimental input.
%   ptheta  Priors of the model
%   htheta  Kernel transformation.
%   pars    Parameters for the MCMC. 
%           pars.verbose   True or false. Defs. False.
%           pars.mc3it      Number of iterations for MC3. Defs. 0.
%           pars.kup        Number of steps for the kernel update. 
%                           Defs. 500.
%           pars.seed       Seed for the random generation. If zero use
%                           rng('shuffle'). Defaults to zero.
%           pars.samples    If true, stores the samples from the chain at 
%                           lowest temperature. Defaults to zero.
%
% Output
%   ps      Samples from the posterior distribuion
%   fe      Free energy estimate.
%
% Uses a standard MCMC on a population and applies an exchange operator
% to improve the mixing.
%
%
% Real parameter evolutionary Monte Carlo with applications to Bayes Mixture
% Models, Journal of American Statistica Association, Liang & Wong 2001.

%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

if ~isfield(pars, 'verbose')
    pars.verbose = 0;
end

if ~isfield(pars, 'mc3it')
    pars.mc3it = 0;
end

if ~isfield(pars, 'kup')
    pars.kup = 500;
end

if ~isfield(pars, 'seed')
    pars.seed = 0;
end

if ~isfield(pars, 'samples')
    pars.samples = 1;
end

if ~isfield(pars, 'thinning')
    pars.thinning = 1;
end

if ~isfield(pars, 'kernel_scale')
    pars.kernel_scale = 0.1 * 0.1 * 0.1;
end

if pars.seed > 0
    rng(pars.seed);
else
    rng('shuffle');
end

if ~isfield(pars, 'T')
    pars.T = linspace(0.1, 1.0, 16).^5;
end

if size(pars.T, 1) > 1
    error('tapas:sem:flat_estimate:input', ...
        'Temperature should be an 1xN matrix');
end

T = pars.T;
nburnin = pars.nburnin;
niter = pars.niter;

% Define the likelihood and prior distributions.
llh = ptheta.llh;
lpp = ptheta.lpp;
sample_priors = ptheta.sample_priors;

nt = numel(T);

% Prepare ptheta
[data] = init_data(data);
htheta = init_htheta(ptheta, htheta);
ptheta = init_ptheta(ptheta, htheta);
otheta = init_theta(data, T, ptheta);

ollh = llh(data, otheta);
olpp = lpp(otheta, ptheta);

ollh = sum(ollh, 1);
nlpp = sum(olpp, 1);

ps_theta = zeros(numel(otheta{end}), niter);
pp_theta = zeros(numel(otheta{end}), niter);
ellh = zeros(nt, niter);
elps = zeros(1, niter);

diagnostics = zeros(1, nt);

ok = init_kernel(otheta, ptheta, htheta);
os = zeros(numel(otheta{1}), nt, pars.kup);

t = 1;

for i = 1 : nburnin + niter
    if i > 1 && mod(i-1, pars.kup) == 0
        diagnostics = diagnostics/pars.kup;
        if pars.verbose
            fprintf(1, '--------------\n')
            fprintf(1, 'Sample: %d\n', i);
            fprintf(1, '--------------\n')
            fprintf(1, 'Acceptance rate:\n')
            fprintf(1, '%0.2f ', diagnostics);
            fprintf(1, '\n');
            fprintf(1, 'Likelihood:\n')
            fprintf(1, '%0.2f ', ollh);
            fprintf(1, '\n');
            if i > nburnin && nt > 1
                fe = trapz(T, mean(ellh(:,1:i-nburnin), 2));
                fprintf(1, 'Fe: %0.05f\n', fe);
            end
        end
        if i <= nburnin
            %ok = update_kernel(t, ok, os, diagnostics, ptheta, htheta);
        end
        diagnostics(:) = 0;
    end
    ntheta = propose_sample(otheta, ptheta, htheta, ok);
    nllh = llh(data, ntheta);

    nllh = sum(nllh, 1);
    nlpp = sum(lpp(ntheta, ptheta), 1);

    v = nllh .* T + nlpp - (ollh .* T + olpp);
    % Reject nas and positive infs
    nansv = isnan(v) | abs(v) == inf; 
    v(nansv) = -inf;

    v = rand(size(v)) < exp(v);

    ollh(v) = nllh(v);
    olpp(v) = nlpp(v);

    otheta(:, v) = ntheta(:, v);

    assert(all(~isnan(ollh)), 'tapas:estimate:ps', ...
        '-inf value in the likelihood');
    
    diagnostics(:) = diagnostics(:) + v(:);

    if i > nburnin
        ps_theta(:, i - nburnin) = otheta{end};
        ellh(:, i - nburnin) = ollh;
        elps(1, i - nburnin) = ollh(end) + olpp(end);
    else
        os(:, :, mod(i-1, pars.kup) + 1) = cell2mat(otheta);
    end

    if nt > 1
    tswap = 1:nt;
    for l = 1:pars.mc3it
        s = ceil(rand() * ( nt - 1));
        p = exp(ollh(s) * T(s+1) + ollh(s+1) * T(s) ...
            - ollh(s) * T(s) - ollh(s+1) * T(s+1));
        if rand() < p
            ollh([s, s+1]) = ollh([s+1, s]);
            olpp([s, s+1]) = olpp([s+1, s]);
            tswap([s, s+1]) = tswap([s+1, s]);
        end
    end
    otheta(:, :) = otheta(:, tswap);
    end
end

% If only one chain don't compute the free energy
if nt > 1
    fe = trapz(T, mean(ellh, 2));
else
    fe = nan;
end

% =============================================================================

% Expected posterior of theta

ptrans = @(x) x; % For a future release. 
ps.pE = mean(ptrans(ps_theta), 2);
ps.pP = mean(ptrans(pp_theta), 2);

% MAP
[~, i] = max(elps(1, :));
ps.map = ptrans(ps_theta(:, i));
% Posteriors of theta
ps_theta = mat2cell(ps_theta, size(ps_theta, 1), ...
    ones(1, size(ps_theta, 2)));
ps.llh = [];
ps.ps_theta = [];
if pars.samples
    ps.ps_theta = ps_theta(1:pars.thinning:end);
    ps.llh = ellh;
end
% Free energy
ps.fe = fe;
% Log likelihood of posterior
% Initial values
ps.data = data;
ps.ptheta = ptheta;
ps.htheta = htheta;
ps.pars = pars;

end

function [data] = init_data(data)
% Sort the data

y = data.y;
u = data.u;

[~, indexes] = sort(y.t);

y.t = y.t(indexes);
y.a = y.a(indexes);
y.i = y.i(indexes);

u.tt = u.tt(indexes);
u.b = u.b(indexes);

data = struct('y', y, 'u', u);

end

function [nk] = init_kernel(theta, ptheta, htheta)
%% Initilize the kernel or covariance matrix of the proposal distribution.
%
% See Exploring an adaptative Metropolis Algorithm
% 

s = htheta.kernel_scale;

njm = tapas_zeromat(ptheta.jm);

c = njm' * htheta.pk * njm;
c = c/eigs(c, 1);
c = chol(c);

nk = cell(numel(theta), 1);
nk(:) = {c};

k =  s * chol(htheta.pk)' * ptheta.jm;
tk = cell(numel(theta), 1);
tk(:) = {k};
nk = struct('S', nk, 's', s, 'k', tk);

end

function [ntheta] = init_theta(data, T, ptheta)
%% Initial values of theta with the prior expected value
%
% Input 
%   T - Temperatures
%   ptheta - Priors
% 
% Output
%   ntheta -- Initialized structure array.

llh = ptheta.llh;
lpp = ptheta.lpp;
sample_priors = ptheta.sample_priors;

if isfield(ptheta, 'p0')
    mu = ptheta.p0;
else
    mu = ptheta.mu;
end

otheta = cell(1, numel(T));
otheta(:) = {mu};

% Make sure that the initilization has positive probability
v = zeros(1, numel(T));
ntheta = otheta;
for i = 1:500
    for j = 1:numel(T)
        if v(j)
            continue
        end
        ntheta{j} = sample_priors(ptheta);
    end
    nllh = llh(data, ntheta);

    nllh = sum(nllh, 1);
    nlpp = sum(lpp(ntheta, ptheta), 1);
    
    v = (nllh + nlpp > -inf); 
    otheta(:, logical(v)) = ntheta(:, logical(v));  
    if all(v)
        break
    end
end

if any(nllh + nlpp == -inf)
    error('tapas:estimate', ...
        'Could not start the algorithm with positive likelihood');
end

fprintf(1, 'Initilize after %d samples from prior\n', i);
ntheta = otheta;

end %

function [nhtheta] = init_htheta(ptheta, htheta)

[np] = size(ptheta.jm, 1);

nhtheta = htheta;
if ~isfield(htheta, 'pk')
    htheta.pk = eye(ptheta.ndims);
end
% TODO
np = np/size(htheta.pk, 1);

nhtheta.pk = kron(eye(np), htheta.pk);

% It is better not to adapt certain parameteres
if ~isfield(htheta, 'mixed')
    nhtheta.mixed = ones(size(ptheta.jm, 1), 1);
else
    nhtheta.mixed = kron(ones(np, 1), logical(htheta.mixed));
end

end


function [nptheta] = init_ptheta(ptheta, htheta)
% Precompute certain quantities

nptheta = ptheta;

if ~isfield(ptheta, 'sm')
    nptheta.sm = tapas_zeromat(ptheta.jm);
end


if isfield(nptheta, 'prepare')
    nptheta = ptheta.prepare(nptheta);
end

end


function [nk] = update_kernel(t, ok, os, ar, ptheta, htheta)
%% Computes a new kernel or covariance for the proposal distribution.
%
% ok    Old kernel
% os    Old samples
% ar    Acceptance rate
%
% See Exploring an adaptative Metropolis Algorithm
% 

t = 3;
c0 = 1.0;
c1 = 0.8;

gammaS = t^-c1;
gammas = c0 * gammaS; 

ns = size(os, 3);
nd = size(os, 1);
nk = ok;

% Optimal log rejection rate
ropt = 0.234;

sm = ptheta.sm;

for i = 1:numel(ok)
    % From Cholesky form to covariance form
    tS = ok(i).S;
    tC = tS' * tS;
    % Empirical variance
    ts = squeeze(os(:, i, :));
    ts = bsxfun(@minus, ts, mean(ts, 2));
    ts = sm' * ts;
    ek = (ts * ts')./(ns - 1);
    % Set new kernel
    nC = tC + gammaS * ( ek - tC );
    % Compute the Cholesky decomposition 
    if ar(i) < 0.009
        nk(i).S = tS; 
        nk(i).s = ok(i).s / 2;
        nk(i).k = ptheta.jm * sqrt(nk(i).s) * tS';
        % Don't mixed this states.
        nk(i).k(~htheta.mixed, :) = ok(i).k(~htheta.mixed, :);
        nk(i).k = sparse(nk(i).k);
        continue
    end 
    try
        nk(i).S = chol(nC);
    catch
        warning('Cholesky decomposition failed.')
        nk(i).S = tS; 
        nk(i).s = ok(i).s / 2;
        nk(i).k = sparse(ptheta.jm * sqrt(nk(i).s) * nk(i).S');
        % Don't mixed these states.
        nk(i).k(~htheta.mixed, :) = ok(i).k(~htheta.mixed, :);
        nk(i).k = sparse(nk(i).k);
        continue
    end
    % Set new scaling
    nk(i).s = exp(log(ok(i).s) + gammas * (ar(i) - ropt));
    nk(i).k = ptheta.jm * sqrt(nk(i).s) * nk(i).S';
    % Don't mixed this states.
    nk(i).k(~htheta.mixed, :) = ok(i).k(~htheta.mixed, :);
    nk(i).k = sparse(nk(i).k);
end
    
end

function [ntheta] = propose_sample(otheta, ptheta, htheta, kn)
%% Draws a new sample from a Gaussian proposal distribution.
%
% Input
%   op          Old parameters
%   ptheta      Prior
%   theta       Hyperpriors
%   kn          Kernels. Two fields: s which is a scaling factor and S which
%               is the Cholosvky decomposition of the kernel.
%
% Ouput
%   ntheta      New output 
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

if nargin < 4
    s = cell(numel(otheta, 1));
    s{:} = 1;
    S = cell(numel(otheta, 1));
    S{:} = eye(numel(otheta{1}));
    kn = struct('S', S, 's', s);
end

nt = numel(otheta);
ntheta = cell(size(otheta));

% Sample and project to a possibly higher dimensional space
for i = 1:nt
    rprop = randn(size(kn(i).k, 2), 1);
    rprop = kn(i).k * rprop;
    ntheta{i} = full(otheta{i} + rprop);
end
 
end
