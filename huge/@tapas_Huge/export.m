function [ listDcms, listConfounds ] = export( obj )
% Export results and data from HUGE object to SPM's DCM format.
% 
% INPUTS:
%   obj - A tapas_Huge object containing data.
%
% OUTPUTS:
%   dcms      - A cell array containing DCM structs in SPM's DCM format.
%   confounds - An array containing group-level confounds (like age or
%               sex) if available. 'confounds' is an array with one row
%               per subject.
% 
% EXAMPLES:
%   [dcms] = EXPORT(obj)    Export fMRI time series and estimation results
%       stored in obj. 
%
%   [dcms,confounds] = EXPORT(obj)    Also export group-level confounds
%       (like age or sex).
% 
% See also tapas_Huge.IMPORT
% 

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2019 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <https://www.gnu.org/licenses/>.
% 
% This software is provided "as is", without warranty of any kind, express
% or implied, including, but not limited to the warranties of
% merchantability, fitness for a particular purpose and non-infringement.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is under active development.
% Considerable changes may occur in future releases. For support please
% refer to:
% https://github.com/translationalneuromodeling/tapas/issues
% 



assert(~isempty(obj.posterior) || ~isempty(obj.model), ...
    'TAPAS:HUGE:export:emptyModel', ...
    'To use export, invert the model or generate simulated data first. ');

% reserve memory
listDcms = repmat({obj.dcm}, obj.N, 1);
listConfounds = [obj.data(:).confounds]';

% prepare posterior
method = [];
if ~isempty(obj.model)
    method = 'GT';
end
if ~isempty(obj.posterior)
    method = obj.posterior.method;
end

switch method
    case 'VB'
        m_k = obj.posterior.m;
        S_k = bsxfun(@rdivide, obj.posterior.S, ...
            permute(obj.posterior.nu, [3 2 1]) - obj.idx.P_c - 1);
        mu_n = obj.posterior.mu_n;
        Sigma_n = obj.posterior.Sigma_n;
        lambda_bar = obj.posterior.b./obj.posterior.a;
        mu_h = obj.prior.mu_h;
        Sigma_h = obj.prior.Sigma_h;
        q_nk = obj.posterior.q_nk;
    case 'MH'
        m_k = obj.posterior.mean.mu;
        S_k = zeros(obj.idx.P_c, obj.idx.P_c, obj.K);
        mu_n = [obj.posterior.mean.theta_c, obj.posterior.mean.theta_h];
        Sigma_n = zeros(obj.idx.P_c + obj.idx.P_h, ...
            obj.idx.P_c + obj.idx.P_h, obj.N);
        try % estimate covariances from samples
            theta = ...
                [reshape([obj.trace.smp(:).theta_c], obj.N, obj.idx.P_c, []), ...
                 reshape([obj.trace.smp(:).theta_h], obj.N, obj.idx.P_h, [])];
            for n = 1:obj.N
                Sigma_n(:,:,n) = cov(permute(theta(n,:,:), [3 2 1]));
            end
            mu = reshape([obj.trace.smp(:).mu], obj.K, obj.idx.P_c, []);
            for k = 1:obj.K
                S_k(:,:,k) = cov(permute(mu(k,:,:), [3 2 1]));
            end
        catch % report marginal variance
            for n = 1:obj.N
                Sigma_n(:,:,n) = diag(...
                    [obj.posterior.variance.theta_c(n,:), ...
                     obj.posterior.variance.theta_h(n,:)]);
            end
            for k = 1:obj.K
                S_k(:,:,k) = diag(obj.posterior.variance.mu(k,:));
            end
        end
        lambda_bar = obj.posterior.mean.lambda;
        mu_h = obj.prior.mu_h;
        Sigma_h = obj.prior.Sigma_h;
        q_nk = obj.posterior.q_nk;
    case 'GT'
        m_k = obj.model.mu_k;
        S_k = obj.model.Sigma_k;
        mu_n = [obj.model.theta_c, obj.model.theta_h];
        Sigma_n = zeros(obj.idx.P_c + obj.idx.P_h, ...
            obj.idx.P_c + obj.idx.P_h, obj.N);
        lambda_bar = obj.model.lambda;
        mu_h = obj.model.mu_h;
        Sigma_h = obj.model.Sigma_h;
        q_nk = obj.model.d;
end


% write SPM DCM structures
for n = 1:obj.N
    
    dcm = listDcms{n};
    
    % export fMRI related data
    dcm.Y.dt = obj.data(n).tr;
    dcm.Y.y = obj.data(n).bold;
    dcm.Y.name = obj.labels.regions;
    if ~isempty(obj.data(n).X0)
        dcm.Y.X0 = obj.data(n).X0;
    end
    dcm.TE = obj.data(n).te;
    
    % export inputs
    dcm.U = obj.inputs(n);
    dcm.U.u = sparse(dcm.U.u);
    dcm.U.name = obj.labels.inputs;

    % export posterior
    if ~isempty(method)
        [~, k] = max(q_nk(n,:));
        dcm = spm_post( dcm, obj, m_k(k,:), S_k(:,:,k), mu_n(n,:), ...
            Sigma_n(:,:,n), lambda_bar(n,:), mu_h, Sigma_h);
    end
       
    listDcms{n} = dcm;

end

end


function [ dcm ] = spm_post( dcm, obj, m_k, S_k, mu_n, Sigma_n, lambda, mu_h, Sigma_h)
%SPM_POST format HUGE posterior into SPM DCM struct
%   subject-level posterior -> dcm.Ep, dcm.Cp and dcm.Ce
%   cluster-level posterior -> dcm.M.pE and dcm.M.pC

    % subject-level
    [A, B, C, D, tau, kappa, epsilon] = obj.theta2abcd(mu_n, obj.idx,...
         obj.R, obj.L);
    % transform C matrix
    if obj.options.nvp.transforminput
        C = .5*exp(C);
    end
    dcm.Ep = struct('A', A, 'B', B, 'C', C, 'D', D, 'transit', tau, ...
        'decay', kappa, 'epsilon', epsilon);
    idx = [obj.idx.clustering; obj.idx.homogenous];
    dcm.Cp = sparse(obj.idx.P_f, obj.idx.P_f);        
    dcm.Cp(idx, idx) = Sigma_n;
    % posterior noise variance
    dcm.Ce = lambda';

    % cluster-level
    [A, B, C, D, tau, kappa, epsilon] = obj.theta2abcd([m_k mu_h],...
         obj.idx, obj.R, obj.L);
    % transform C matrix
    if obj.options.nvp.transforminput
        C = .5*exp(C);
    end
    dcm.M.pE = struct('A', A, 'B', B, 'C', C, 'D', D, 'transit', tau, ...
        'decay', kappa, 'epsilon', epsilon);

   % mean cluster covariance
    pC = zeros(obj.idx.P_f);
    pC(obj.idx.clustering, obj.idx.clustering) = S_k;
    pC(obj.idx.homogenous, obj.idx.homogenous) = Sigma_h;
    dcm.M.pC = sparse(pC);

end

