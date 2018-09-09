function [llh] = tapas_mdlinear_hier_llh(data, theta, ptheta)
%% Likelihood of the nodes of a linear multivariate model with diagonal 
% precisions on each parameter. 
%
% It is the likelihood of 
%
% p(y|x, beta, pi) = N(y - x*b, 1/pi)
%
% Note that this is a multivariate regression problem, so beta is a matrix
% in which each column correspond to one parameter of the model.
%

%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

y = data.y;
u = data.u;

x = data.u.x;
theta = theta.y;

[np, nc] = size(y);

llh = zeros(np, nc);

ln2pi = log(2 * pi);

for j = 1:nc
    % Mean
    beta = theta{j}.mu;
    % Precision
    pe = theta{j}.pe;
    % Regressors

    lpe = log(pe);

    % number of subjects x number of parameters
    py = u.x * beta;
    [ins, inp] = size(py);
    % TODO this is a problem because it is not consistent
    ty = cell2mat(y(:, j));
    
    % Residuals
    r = (reshape(ty, inp, ins)' - py);
    
    % Prediction error 
    e = sum(bsxfun(@times, pe, r .* r), 2);
    
    % This is a vector. For each subject needs to include the precision.
    llh(:, j) = sum(-0.5 * ln2pi +  0.5 * lpe) - 0.5 * e;
end


end

