function [ nfe ] = vb_nfe( obj )
% Evaluate negative free energy for HUGE model given current parameter
% estimates.
% 
% This is a protected method of the tapas_Huge class. It cannot be called
% from outside the class.
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


%% eval
q_plus_nu = obj.aux.q_k + obj.prior.nu_0;
% weights
F_on = (obj.aux.q_k + obj.prior.alpha_0 - obj.posterior.alpha)'* ...
    psi(0, obj.posterior.alpha);

F_on = F_on + sum(gammaln(obj.posterior.alpha));
% labels
F_on = F_on - sum(obj.posterior.q_nk(:).*log(obj.posterior.q_nk(:) + ...
    realmin));
% clusters
F_on = F_on -.5*obj.idx.P_c*sum((obj.aux.q_k + obj.prior.tau_0)./...
    obj.posterior.tau + log(obj.posterior.tau));

F_on = F_on + obj.K*obj.idx.P_c*(obj.idx.P_c - 1)/4*log(pi) + ... %%% -> off
    sum(sum(gammaln(bsxfun(@minus, obj.posterior.nu, 0:obj.idx.P_c-1)/2)));

F_on = F_on +.5*(q_plus_nu - obj.posterior.nu)'* ...
    sum(psi(0, bsxfun(@minus, obj.posterior.nu, 0:obj.idx.P_c-1)/2), 2);

dm = permute(bsxfun(@minus, obj.posterior.m, obj.prior.m_0), [3 2 1]);
dm = permute(dm, [2 1 3]).*sum(bsxfun(@times, obj.aux.nu_inv_S, dm), 2);
F_on = F_on -.5*obj.prior.tau_0*sum(dm(:));

F_on = F_on -.5*sum(sum(sum(bsxfun(@times, obj.aux.nu_inv_S, obj.prior.S_0))));

F_on = F_on -.5*sum(q_plus_nu.*obj.aux.ldS);
% dcm
Sigma_n_c = obj.posterior.Sigma_n(1:obj.idx.P_c,1:obj.idx.P_c,:);
mu_n_c = obj.posterior.mu_n(:,1:obj.idx.P_c);
if obj.options.confVar == 1
    Sigma_n_c = Sigma_n_c + obj.aux.Sigma_beta;
    mu_n_c = mu_n_c - obj.aux.mu_beta;
end
for k = 1:obj.K
    if obj.options.confVar > 1
        Sigma_n_c = obj.posterior.Sigma_n(1:obj.idx.P_c,1:obj.idx.P_c,:) + ...
            obj.aux.Sigma_beta(:,:,:,k);
        mu_n_c = obj.posterior.mu_n(:,1:obj.idx.P_c) - ...
            obj.aux.mu_beta(:,:,k);
    end
    dmu = permute(bsxfun(@minus, mu_n_c, obj.posterior.m(k,:)), [3 2 1]);
    tmp = Sigma_n_c + bsxfun(@times, dmu, permute(dmu, [2 1 3]));
    F_on = F_on -.5*obj.posterior.q_nk(:,k)'*squeeze(sum(sum( ...
        bsxfun(@times, obj.aux.nu_inv_S(:,:,k), tmp), 1), 2));
end

dmu_h = permute(bsxfun(@minus, obj.posterior.mu_n(:, obj.idx.P_c+1:end), ...
    obj.prior.mu_h), [3 2 1]);
Sigma_h = sum(obj.posterior.Sigma_n(obj.idx.P_c+1:end, obj.idx.P_c+1:end, :) + ...
    bsxfun(@times, dmu_h, permute(dmu_h, [2 1 3])), 3);
F_on = F_on -.5*sum(sum(obj.aux.Pi_h.*Sigma_h));

F_on = F_on +.5*sum(obj.aux.ldSigma);

F_on = F_on -.5*(obj.aux.b_prime(:)'*obj.aux.lambda_bar(:)) + ...
    obj.posterior.a(:)'*(1 - log(obj.posterior.b(:)) - obj.prior.b_0./...
    obj.posterior.b(:));
% confounds
if obj.options.confVar
    dmu_beta = reshape(obj.posterior.m_beta, obj.M, []);
    dmu_beta = permute(bsxfun(@minus, dmu_beta, obj.prior.m_beta_0), [1 3 2]);
    S_beta = sum(bsxfun(@times, dmu_beta, permute(dmu_beta, [2 1 3])) + ...
        reshape(obj.posterior.S_beta, obj.M, obj.M, []), 3);
    F_on = F_on -.5*sum(sum(obj.aux.Pi_beta_0.*S_beta));
    
    F_on = F_on +.5*sum(obj.aux.ldS_beta(:));
end

nfe = F_on + obj.aux.F_off;


end

