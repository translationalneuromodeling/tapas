function [llh, ny] = mpdcm_fmri_llh(y, u, theta, ptheta, sloppy)
%% Computes the likelihood of the data.
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

if nargin < 5
    sloppy = 0;
end

% Integrates the system

ny = mpdcm_fmri_int(u, theta, ptheta, sloppy);

% Computes the likelihood

s1 = size(theta, 1);
s2 = size(theta, 2);

llh = zeros(size(theta));

l2p = log(2*pi);

for i = 1:s1
    y0 = y{s1}';
    y0 = y0(:);
    for j = 1:s2
        theta0 = theta{(i-1)*s1 + j};
        % Check the eigen values
        %ev = eigs(theta0.A, 1);

        %if ~all(isreal(ev)) || max(ev) > 0 
        %    llh((i-1)*s1+j) = -inf;
        %end

        ny0 = ny{(i-1)*s1 + j}(:);

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

            llh((i-1)*s1+j) = -numel(e) * l2p  - 2*sum(log(diag(sQ))) ...
                - 0.5*e'*(nQ\e);
        else
            nQ = zeros(size(ptheta.dQ.Q{1}));
            for k = 1:numel(ptheta.dQ.Q)
               nQ = nQ + tlambda(k)*ptheta.dQ.Q{k};
            end
            nQ = nQ(:);
            llh((i-1)*s1+j) = -numel(e) * l2p  - sum(log(nQ)) ...
                - 0.5*e'*((1./nQ).*e);
        end
    end
end

end
