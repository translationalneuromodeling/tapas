function [theta, ny, nllh, nlpp] = tapas_mpdcm_fmri_sample_gibbs(y, u, ...
    theta, ptheta, pars, ny, nllh, nlpp)
%% Makes a Gibbs step of a set of the parameters.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

% Perfom a gibb step

if numel(ptheta.X0) 
    [theta, dlpp] = draw_samples(y, u, theta, ptheta, ny, pars);
    nllh = pars.fllh(y, u, theta, ptheta, ny);
    nllh = sum(nllh, 1);
    nlpp = nlpp + dlpp;

    assert(all(-inf < nllh), 'mpdcm:fmri:ps', ...
        'gibbs -inf value in the likelihood');
end

end % 

function [theta, dlpp] = draw_samples(y, u, theta, ptheta, ny, pars)
%% Draw a sample for lthe distribution

nt = numel(theta);
nr = size(y{1}, 1);
nb = size(ptheta.X0, 2);

I =  eye(nb)/ptheta.X0_variance;
y = y{1};
ops0 = struct('UT', true);
ops = struct('TRANSA', true, 'UT', true);
dlpp = zeros(1, nt);
for i = 1:nt
    % Assume no correlations between regions i.e., treat the problem as massive
    % multivariate problem

    %if any(isnan(ny{i}(:))) || any(isinf(ny{i}(:)))
    %    continue
    %end
    % Fit only the residual
    r = y' - ny{i};
    %beta = ptheta.omega * (y' - ny{i});
    lambda = exp(theta{i}.lambda) * pars.T(i);
    tc = ptheta.X0' * r;
    dlpp(i) = 0.5 * 1/ptheta.X0_variance * ...
        sum(sum(theta{i}.beta .* theta{i}.beta ));
    for j = 1:nr
        cd = chol(ptheta.X0X0 + I./lambda(j));
        beta = linsolve(cd, linsolve(cd, tc(:, j), ops0), ops);
        theta{i}.beta(:, j) = beta + ...
            linsolve(cd, sqrt(1/lambda(j)) * randn(nb, 1), ops);
    end

    dlpp(i) = dlpp(i) - 0.5 * 1/ptheta.X0_variance * ...
        sum(sum(theta{i}.beta .* theta{i}.beta ));

end

end %

