function [ps, fe] = tapas_ti_estimate(y, u, ptheta, htheta, pars)
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


T = pars.T;
nburnin = pars.nburnin;
niter = pars.niter;

nt = numel(T);

llh = @ptheta.llh;
lpp = @ptheta.lpp;

% Prepare ptheta

[ptheta] = pars.init_ptheta(ptheta, htheta, pars);
[htheta] = pars.init_htheta(ptheta, htheta, pars);
[otheta] = pars.init_theta(ptheta, htheta, pars);
[u] = pars.init_u(u, ptheta, pars);
[y] = pars.init_y(y, ptheta, pars);

[ollh, ox] = llh(y, [], u, otheta, ptheta);
[olpp] = lpp(y, ox, u, otheta, ptheta);

ollh = sum(ollh, 1);
nlpp = sum(olpp, 1);

ps_theta = zeros(numel(otheta{end}), niter);
pp_theta = zeros(numel(otheta{end}), niter);
ellh = zeros(nt, niter);

diagnostics = zeros(1, nt);

os = zeros(numel(otheta{1}), nt, pars.kup);

t = 1;

for i = 1 : nburnin + niter

    if i > 1 && mod(i-1, pars.kup) == 0
        diagnostics = diagnostics/pars.kup;
        if pars.verbose
            fprintf(1, 'Iter %d, diagnostics:  ', i);
            fprintf(1, '%0.2f ', diagnostics);
            fprintf(1, '\n');
            fprintf(1, '%0.2f ', ollh);
            fprintf(1, '\n');
            fprintf(1, '%0.2f ', olpp);
            fprintf(1, '\n');
            if i > nburnin && nt > 1
                fe = trapz(T, mean(ellh(:, 1 : i - nburnin), 2));
                fprintf(1, 'Fe: %0.05f\n', fe);
            end
        end
        if i <= nburnin
            % Keep changing the kernel at the same rate
            htheta.ok = update_kernel(1, os, diagnostics, ptheta, htheta);
        end
        diagnostics(:) = 0;
        t = t + 1;
    end

    [ox, otheta, ollh, olpp, v] = tapas_ti_step_mh(y, u, ox, otheta, ...
        ollh, olpp, ptheta, htheta, pars);

    diagnostics(:) = diagnostics(:) + v(:);

    if i > nburnin
        ps_theta(:, i - nburnin) = otheta{end};
        ellh(:, i - nburnin) = ollh;
    else
        os(:, :, mod(i-1, pars.kup) + 1) = cell2mat(otheta);
    end

    for l = 1:pars.mc3it
        s = ceil(rand()*(nt-1));
        p = min(1, exp(ollh(s) * T(s+1) + ollh(s+1) * T(s) ...
            - ollh(s) * T(s) - ollh(s+1) * T(s+1)));
        if rand() < p
            ollh([s, s+1]) = ollh([s+1, s]);
            olpp([s, s+1]) = olpp([s+1, s]);
            ox([s, s+1]) = ox([s+1, s]);
            otheta(:, [s, s+1]) = otheta(:, [s+1, s]); 
        end
    end

end

% If only one chain don't compute the free energy
if nt > 1
    fe = trapz(T, mean(ellh, 2));
else
    fe = Nan;
end

% =============================================================================

% Expected posterior of theta

ptrans = ptheta.ptrans; % Transformation function of the parameters
ps.pE = mean(ptrans(ps_theta), 2);
ps.pP = mean(ptrans(pp_theta), 2);

% MAP
[~, i] = max(ellh(end, :));
ps.map = ptrans(ps_theta(:, i));
% Posteriors of theta
ps.ps_theta = [];
ps.llh = [];
if pars.samples
    ps.ps_theta = ps_theta;
    ps.llh = ellh;
end
% Free energy
ps.F = fe;
% Log likelihood of posterior
% Initial values
ps.y = y;
ps.u = u;
ps.ptheta = ptheta;
ps.htheta = htheta;
ps.pars = pars;

end % tapas_ti_estimate

function [nk] = update_kernel(t, os, ar, ptheta, htheta)
%% Computes a new kernel or covariance for the proposal distribution.
%
% ok    Old kernel
% os    Old samples
% ar    Acceptance rate
%
% See Exploring an adaptative Metropolis Algorithm
% 

c0 = 1.0;
c1 = 0.8;

gammaS = t^-c1;
gammas = c0*gammaS; 

ns = size(os, 3);
nd = size(os, 1);
ok = htheta.ok;
nk = ok;

% Optimal log rejection rate
ropt = 0.234;

sm = ptheta.sm;

for i = 1:numel(ok)
    % From Cholesky form to covariance form
    ok(i).S = ok(i).S * ok(i).S';
    % Empirical variance
    ts = squeeze(os(:, i, :));
    % Prevent overflow error
    ts = bsxfun(@minus, ts, sum(ts/size(ts, 2), 2));
    ts = sm' * ts;
    ek = (ts * ts')./(ns-1);
    % Set new kernel
    nk(i).S = ok(i).S + gammaS * ( ek - ok(i).S);
    % Compute the Cholesky decomposition 
    try
        nk(i).S = chol(nk(i).S);
    catch
        %warning('Cholesky decomposition failed.')
        nk(i).s = nk(i).s / 2;
        nk(i).k = ptheta.jm * nk(i).s * nk(i).S;
        continue
    end
    % Set new scaling
    nk(i).s = exp(log(ok(i).s) + gammas * (ar(i) - ropt));
    nk(i).k = ptheta.jm * nk(i).s * nk(i).S; 
end
    
end

