function [q, ntheta] = mpdcm_fmri_gmodel(y, u, theta, ptheta) 
%% Estimate the model using an approximate variational Bayes method
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

[y, u, theta, ptheta] = mpdcm_fmri_qinput(y{1}, u{1}, theta{1}, ptheta);

y = {y};
u = {u};
theta = {theta};

ns = size(y{1}, 2);
nr = size(y{1}, 1);
np = numel(ptheta.mtheta);

assert(nr > 1, 'mpdcm:fmri:gmodel:input', ...
    'Single region models are not implemented');

q = theta{1}.q;

% Optimize the posterior
for i = 1:10
    theta{1}.lambda = log(q.lambda.a) - log(q.lambda.b);
    [mu, ny, dfdx] = mpdcm_fmri_map(y, u, theta, ptheta);

    theta = mpdcm_fmri_set_parameters(mu, theta, ptheta);

    q.theta.mu = mu;
    % Approximate hessian

    % Weight the hessian by the precisions

    dfdx{1} = dfdx{1}(:,:,1:np-nr);

    h1 = bsxfun(@times, dfdx{1}, reshape(exp(theta{1}.lambda), 1, nr, 1));
    h1 = reshape(h1, ns*nr, 1, np-nr);
    h1 = squeeze(h1);

    h2 = reshape(dfdx{1}, ns*nr, 1, np-nr);
    h2 = squeeze(h2);

    q.theta.pi = 0.5*h1'*h2; 
    
    % Estimate the noise

    q.lambda.a(:) = 0.5*ns;% + ptheta.p.lambda.a - 1;

    e = ny{1} - y{1}';
    for k = 1:nr
        h = dfdx{1}(:, k, :);
        h = squeeze(h);
        h = 0.5*h'*h; 
        q.lambda.b(k) = 0.5*e(:,k)'*e(:,k) + trace(q.theta.pi\h) + ...
            ptheta.p.lambda.b(k);
        q.lambda.b(k) = 0.5*e(:,k)'*e(:,k);
    end
    %q.lambda.b = 1./q.lambda.b;
end

theta{1}.lambda = log(q.lambda.a) - log(q.lambda.b);

ntheta = theta;
end
