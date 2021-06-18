function [ output ] = tapas_rdcm_store_parameters(DCM, mN_cut, sN, aN, bN, logF, logF_term, idx_x, z_cut, args)
% [ output ] = tapas_rdcm_store_parameters(DCM, mN_cut, sN, aN, bN, logF, logF_term, idx_x, z_cut, args)
% 
% Wraps parameters to the output structure.
% 
%   Input:
%   	DCM             - model structure
%       mN_cut          - posterior means of connectivity parameters
%       sN              - posterior covaraince of connectivity parameters
%       aN              - posterior shape parameter of measurement noise
%       bN              - posterior rate parameter of measurement noise
%       logF            - negative free energy
%       logF_term       - seperate terms of the negative free energy
%       idx_x           - regressor/parameter index
%       z_cut           - posterior binary indicators
%       args            - arguments
%
%   Output:
%       output          - output structure
%
 
% ----------------------------------------------------------------------
% 
% Authors: Stefan Fraessle (stefanf@biomed.ee.ethz.ch), Ekaterina I. Lomakina
% 
% Copyright (C) 2016-2021 Translational Neuromodeling Unit
%                         Institute for Biomedical Engineering
%                         University of Zurich & ETH Zurich
%
% This file is part of the TAPAS rDCM Toolbox, which is released under the 
% terms of the GNU General Public License (GPL), version 3.0 or later. You
% can redistribute and/or modify the code under the terms of the GPL. For
% further see COPYING or <http://www.gnu.org/licenses/>.
% 
% Please note that this toolbox is in an early stage of development. Changes 
% are likely to occur in future releases.
% 
% ----------------------------------------------------------------------


% remove confound regressors
Nc = size(DCM.U.X0,2);
DCM.b = DCM.b(:,:,1:end-Nc);
DCM.c = DCM.c(:,1:end-Nc);

% get number of regions and inputs
[nr, nu] = size(DCM.c);

% get the posterior parameter estimates
if min(size(idx_x)) == 1
    mN = zeros(nr,length(idx_x));
    mN(repmat(idx_x,nr,1)) = mN_cut;
else
    mN = mN_cut;
end


%% mean

% get mean for connectivity parameters
output.Ep          = tapas_rdcm_empty_par(DCM);
output.Ep.A        = mN(1:nr,1:nr);
output.Ep.B        = reshape(mN(1:nr,nr+1:nr+nr*nu),[nr nr nu]);
output.Ep.C        = mN(1:nr,end-Nc-nu+1:end-Nc);
output.Ep.baseline = mN(1:nr,end-Nc+1:end);

% modify driving inputs
if ( strcmp(args.type,'r') )
    output.Ep.C        = output.Ep.C*16;
end


%% variance

% get the number of potential connections
D = numel(output.Ep.A) + numel(output.Ep.C);

% create the covariance matrix (for computational reasons, only recommended for small DCMs)
if ( isempty(z_cut) && args.evalCp == 1 )
    
    % empty covariance matrix
    output.Cp = sparse(D,D);
    
    % get the number of entries
    nr_A    = numel(output.Ep.A);
    nr_C    = numel(output.Ep.C);
    nr_A_B  = numel(output.Ep.A(1,:))+numel(output.Ep.B(1,:,:))+numel(output.Ep.B(1,:,1));
    
    
    % define an index matrix for endogenous connections
    indexA = reshape(1:nr_A,nr,nr);
    
    % get the (co)variances of the endogenous connections
    for k = 1:nr
        for int = 1:nr
            for int2 = 1:nr
                output.Cp(indexA(k,int),indexA(k,int2)) = sN{k}(int,int2);
            end
        end
    end
    
    % define an index matrix for driving input parameters
    indexC = reshape(1:nr_C,nr,nu);
    
    % get the (co)variances of the driving input parameters
    for k = 1:nr
        for int = 1:nu
            for int2 = 1:nu
                output.Cp(nr_A+indexC(k,int),nr_A+indexC(k,int2)) = sN{k}(nr_A_B+int,nr_A_B+int2);
            end
        end
    end
    
    % get the covariances between endogeous and driving input parameters
    for k = 1:nr
        for int = 1:nr
            for int2 = 1:nu
                output.Cp(indexA(k,int),nr_A+indexC(k,int2)) = sN{k}(int,nr_A_B+int2);
                output.Cp(nr_A+indexC(k,int2),indexA(k,int)) = sN{k}(int,nr_A_B+int2);
            end
        end
    end
end


% store the regions-wise posterior covariance matrices
output.sN = sN;



%% precision

% store precision parameters
output.t  = aN./bN;
output.aN = aN;
output.bN = bN;


%% connection probabilities

% store connection probabilities
if ( ~isempty(z_cut) )
    if min(size(idx_x)) == 1
        z = zeros(nr,length(idx_x));
        z(repmat(idx_x,nr,1)) = z_cut;
    else
        z = z_cut;
    end
    
    output.Ip   = tapas_rdcm_empty_par(DCM);
    output.Ip.A = z(1:nr,1:nr);
    output.Ip.B = reshape(z(1:nr,nr+1:nr+nr*nu),[nr nr nu]);
    output.Ip.C = z(1:nr,end-nu:end-1);
else
    output.Ip   = tapas_rdcm_empty_par(DCM);
    output.Ip.A = DCM.a;
    output.Ip.B = DCM.b;
    output.Ip.C = DCM.c;
end


%% free energy

% store free energy
output.logF   = sum(logF);
output.logF_r = logF;

% store the components of the free energy
if ( ~isempty(logF_term) )
    output.logF_term.log_lik        = sum(logF_term.log_lik);
    output.logF_term.log_p_weight   = sum(logF_term.log_p_weight);
    output.logF_term.log_p_prec     = sum(logF_term.log_p_prec);
    output.logF_term.log_p_z        = sum(logF_term.log_p_z);
    output.logF_term.log_q_weight   = sum(logF_term.log_q_weight);
    output.logF_term.log_q_prec     = sum(logF_term.log_q_prec);
    output.logF_term.log_q_z        = sum(logF_term.log_q_z);
end

end