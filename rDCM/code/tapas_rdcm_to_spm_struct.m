function [ DCM ] = tapas_rdcm_to_spm_struct(output)
% [ DCM ] = tapas_rdcm_to_spm_struct(output)
% 
% Converts the output structure from rDCM into a DCM structure that SPM12
% can read.
% 
%   Input:
%       output          - rDCM output structure
%
%   Output:
%       DCM             - DCM structure (for integrability with SPM12)
%
 
% ----------------------------------------------------------------------
% 
% Authors: Stefan Fraessle (stefanf@biomed.ee.ethz.ch), Ekaterina I. Lomakina
% 
% Copyright (C) 2016-2022 Translational Neuromodeling Unit
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


% structure of the DCM
DCM.a = output.Ip.A;
DCM.b = output.Ip.B;
DCM.c = output.Ip.C;
DCM.d = output.Ip.D;

% number of regions and number of data points
DCM.n = size(DCM.a,1);
DCM.v = size(output.signal.y_source,1)/DCM.n;

% empirical data
DCM.Y.y      = reshape(output.signal.y_source,DCM.v,DCM.n);
DCM.Y.yd_fft = reshape(output.signal.yd_source_fft,DCM.v,DCM.n);
DCM.Y.name	 = output.signal.name;
DCM.Y.dt     = output.inputs.dt*16;

% inputs
DCM.U = output.inputs;

% model settings
DCM.M.IS    = output.inversion;
DCM.M.pE.A  = output.priors.m0(1:size(DCM.a,1),1:size(DCM.a,2));
DCM.M.pE.C  = output.priors.m0(1:size(DCM.a,1),end-size(DCM.c,2)+1:end);

% prior covariance matrix
pC.A = 1./output.priors.l0(1:size(DCM.a,1),1:size(DCM.a,2));
pC.C = 1./output.priors.l0(1:size(DCM.a,1),end-size(DCM.c,2):end-1);

% further model settings
DCM.M.pC = diag([pC.A(:); pC.C(:)]);

% options
DCM.options.nonlinear   = 0;
DCM.options.two_state   = 0;
DCM.options.stochastic  = 0;
DCM.options.nograph     = 0;

% error covariance
DCM.Ce = 1./output.t;

% posterior means
DCM.Ep.A = output.Ep.A;
DCM.Ep.C = output.Ep.C;

% get the number of various entries
nr_A    = numel(output.Ep.A);
nr_C    = numel(output.Ep.C);
nr_A_B  = numel(output.Ep.A(1,:))+numel(output.Ep.B(1,:,:))+numel(output.Ep.B(1,:,1));
nr      = size(output.Ep.A,1);
nu      = size(output.Ep.C,2);

% posterior covariance matrix (if full covariance matrix does not exist, 
% the matrix will be constructed from the region-wise posterior covariance 
% matrices; NOTE: For whole-brain models with many region, this may take a
% relatively long time)
try
    DCM.Cp = output.Cp;
    
catch
    
    % get the number of potential connections
    D = numel(DCM.a) + numel(DCM.c);

    % empty covariance matrix
    DCM.Cp = sparse(D,D);
    
    % define helper arrays/cells
    sN_size = NaN(nr,1);
    sN      = cell(nr,1);
    
    % check dimensionality
    for k = 1:nr
        sN_size(k) = size(output.sN{k},1);
    end
    
    % max dimensionality
    sN_maxSize = max(sN_size);
    
    % equalize dimensionality
    for k = 1:nr
        sN{k} = zeros(sN_maxSize,sN_maxSize);
        sN{k}(1:sN_size(k),1:sN_size(k)) = output.sN{k};
    end
    
    % define an index matrix for endogenous connections
    indexA = reshape(1:nr_A,nr,nr);
    
    % get the (co)variances of the endogenous connections
    for k = 1:nr
        for int = 1:nr
            for int2 = 1:nr
                DCM.Cp(indexA(k,int),indexA(k,int2)) = sN{k}(int,int2);
            end
        end
    end
    
    % for task-based fMRI (no resting state)
    if ( ~strcmp(DCM.U.name,'null') )

        % define an index matrix for driving input parameters
        indexC = reshape(1:nr_C,nr,nu);

        % get the (co)variances of the driving input parameters
        for k = 1:nr
            for int = 1:nu
                for int2 = 1:nu
                    DCM.Cp(nr_A+indexC(k,int),nr_A+indexC(k,int2)) = sN{k}(nr_A_B+int,nr_A_B+int2);
                end
            end
        end

        % get the covariances between endogeous and driving input parameters
        for k = 1:nr
            for int = 1:nr
                for int2 = 1:nu
                    DCM.Cp(indexA(k,int),nr_A+indexC(k,int2)) = sN{k}(int,nr_A_B+int2);
                    DCM.Cp(nr_A+indexC(k,int2),indexA(k,int)) = sN{k}(int,nr_A_B+int2);
                end
            end
        end
    end
end

% obtain variables in vectorized form
pE_vec = [DCM.M.pE.A(:); DCM.M.pE.C(:)];
Ep_vec = [DCM.Ep.A(:); DCM.Ep.C(:)];
Cp_vec = diag(DCM.Cp);

% turn off warnings temporarily
warning('off');

% compute the posterior probability of each parameter
T  = full(pE_vec);
Pp = 1 - normcdf(T,abs(Ep_vec),sqrt(Cp_vec));

% find non-existing parameters
nCp = ( ones(size(T)) & ones(size(Ep_vec)) & Cp_vec>0 );
if any(~nCp(:))
    Pp(~nCp) = NaN;
end

% turn on warnings
warning('on');

% asign the posterior probability
DCM.Pp.A = reshape(Pp(1:nr_A),nr,nr);
DCM.Pp.C = reshape(Pp(nr_A+1:end),nr,nu);

% obtain the posterior variances for each region
for k = 1:nr

    % obtain diagonal from the region-wise covariance matrix
    Cpdiag = diag(sN{k});

    % get variances of endogenous parameters
    DCM.Vp.A(k,:) = Cpdiag(1:nr);

    % get the driving input variances for task-based fMRI (no resting state)
    if ( ~strcmp(DCM.U.name,'null') )
        DCM.Vp.C(k,:) = Cpdiag(nr_A_B+1:nr_A_B+nu);
    else
        DCM.Vp.C(k,:) = zeros(1,nu);
    end
    
end

% predicted data and residuals (power spectral density)
try
    DCM.y = abs(reshape(output.signal.yd_pred_rdcm_fft,DCM.v,DCM.n)).^2;
    DCM.R = abs(reshape(output.signal.yd_source_fft,DCM.v,DCM.n)).^2 - DCM.y;
end

% predicted data (temporal derivative in Fourier space)
DCM.yd_fft = output.signal.yd_pred_rdcm_fft;

% negative free energy
DCM.F = output.logF;

end