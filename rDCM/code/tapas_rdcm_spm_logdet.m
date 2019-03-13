function [ H ] = tapas_rdcm_spm_logdet(C)
% returns the log of the determinant of positive semi-definite matrix C
% FORMAT [H] = spm_logdet(C)
% H = log(det(C))
%
% spm_logdet is a computationally efficient operator that can deal with
% sparse matrices
%__________________________________________________________________________
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_logdet.m 4278 2011-03-31 11:48:00Z karl $

sw    = warning('off','MATLAB:log:logOfZero');

% assume diagonal form
%--------------------------------------------------------------------------
TOL   = 1e-16;
n     = length(C);
s     = diag(C);
i     = find(s > TOL & s < 1/TOL);
C     = C(i,i);
H     = sum(log(diag(C)));

% invoke det if non-diagonal
%--------------------------------------------------------------------------
[i j] = find(C);
if any(i ~= j)
      n = length(C);
      a = exp(H/n);
      H = H + log(det(C/a));           
end

% invoke svd if rank deficient
%--------------------------------------------------------------------------
if ~isreal(H) || isinf(H)
    s  = svd(full(C));
    H  = sum(log(s(s > TOL & s < 1/TOL)));
end
H     = full(H);
warning(sw)
