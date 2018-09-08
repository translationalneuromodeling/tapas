function [llh] = tapas_mpdcm_fmri_llh(y, u, theta, ptheta, ny)
%% Computes the likelihood of the data.
%
% Input:
% y         -- Cell array. Each cell contains the experimental data.
% u         -- Cell array. Each cell contains the model input.
% theta     -- Cell array. Each cell contains the model parameters.
% ptheta    -- Structure. Prior of all models.
% ny        -- Cell array. Each cell is a prediction of the data.
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

% Computes the likelihood

s1 = size(ny, 1);
s2 = size(ny, 2);

llh = zeros(size(ny));

l2p = log(2 * pi);

x0 = ptheta.X0;

for i = 1:s1
    y0 = y{i}';
    %y0 = y0(:);
    [sy1, sy2] = size(y0);

    for j = 1:s2

        % Check the eigen values
        %ev = eigs(theta0.A, 1);

        %if ~all(isreal(ev)) || max(ev) > 0 
        %    llh((i-1)*s1+j) = -inf;
        %end

        theta0 = theta{i, j};
        ny0 = ny{i, j};
        
        if numel(x0)
            ny0 = ny0 + x0 * theta0.beta;
        end
        %ny0 = ny0(:);
        
        e = y0 - ny0;

        % Compute the hyperpriors

        tlambda = exp(theta0.lambda)';

        % Optimize if the covariance matrix is diagonal or not.

        if ~ptheta.dQ.dm % It is not diagonal
            nQ = exp(-32) * eye(size(ptheta.Q{1})) + tlambda(1) * ptheta.Q{1};
            for k = 2:numel(ptheta.Q)
                nQ = nQ + tlambda(k) * ptheta.Q{k};
            end

            sQ = chol(nQ);

            llh((i-1)*s1+j) = -0.5 * numel(e) * l2p  - 2*sum(log(diag(sQ)))...
                - 0.5*e'*(nQ\e);
        else % It is diagonal
            %nQ = zeros(size(ptheta.dQ.Q{1}));
            %for k = 1:numel(ptheta.dQ.Q)
               %nQ = nQ + tlambda(k) * ptheta.dQ.Q{k};
            llh(i, j) = -0.5 * sy1 * sy2 * l2p  + 0.5 * sy1 * sum(theta0.lambda) ...
                - 0.5 * sum(sum(e.*e, 1) .* tlambda);

            %end
            if isnan(llh(i, j))
                llh(i, j) = -inf;
            elseif llh(i, j) == inf
                llh(i, j) = -inf;
            end
        end
    end
end

end
