function [llh, ny] = tapas_mpdcm_fmri_llh(y, u, theta, ptheta, sloppy)
%% Computes the likelihood of the data.
%
% Input:
% y         -- Cell array. Each cell contains the experimental data.
% u         -- Cell array. Each cell contains the model input.
% theta     -- Cell array. Each cell contains the model parameters.
% ptheta    -- Structure. Prior of all models.
% sloppy    -- Scalar. If 0 input is not checked. Defaults to 1.
%
% Output:
% llh       -- Cell array. Each cell contains an scalar with the loglikelihood
%           of the correspoding cell.
% ny        -- Cell array. Each cell contains an array of predicted signales.
% 

% aponteeduardo@gmail.com
%
% Author: Eduardo Aponte, TNU, UZH & ETHZ - 2015
% Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
%
% Licensed under GNU General Public License 3.0 or later.
% Some rights reserved. See COPYING, AUTHORS.
%
% Revision log:
%
%

if nargin < 5
    sloppy = 0;
end

% Integrates the system

ny = tapas_mpdcm_fmri_int(u, theta, ptheta, sloppy);

% Computes the likelihood

s1 = size(theta, 1);
s2 = size(theta, 2);

llh = zeros(size(theta));

l2p = log(2*pi);

for i = 1:s1
    y0 = y{i}';
    y0 = y0(:);
    for j = 1:s2

        theta0 = theta{i, j};

        % Check the eigen values
        %ev = eigs(theta0.A, 1);

        %if ~all(isreal(ev)) || max(ev) > 0 
        %    llh((i-1)*s1+j) = -inf;
        %end

        ny0 = ny{i, j}(:);

        e = y0 - ny0;

        % Compute the hyperpriors

        tlambda = exp(theta0.lambda);

        % Optimize if the covariance matrix is diagonal or not.

        if ~ptheta.dQ.dm
            nQ = exp(-32) * eye(size(ptheta.Q{1})) + tlambda(1) * ptheta.Q{1};
            for k = 2:numel(ptheta.Q)
                nQ = nQ + tlambda(k) * ptheta.Q{k};
            end

            sQ = chol(nQ);

            llh((i-1)*s1+j) = -0.5 * numel(e) * l2p  - 2*sum(log(diag(sQ)))...
                - 0.5*e'*(nQ\e);
        else
            nQ = zeros(size(ptheta.dQ.Q{1}));
            for k = 1:numel(ptheta.dQ.Q)
               nQ = nQ + tlambda(k)*ptheta.dQ.Q{k};
            end
            nQ = nQ(:);
            llh(i, j) = -0.5*numel(e) * l2p  + 0.5*sum(log(nQ)) ...
                - 0.5*e'*(nQ.*e);
        end
    end
end

end
