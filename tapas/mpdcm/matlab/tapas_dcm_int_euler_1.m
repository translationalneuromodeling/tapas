function y = tapas_dcm_int_euler_1(dcms)

n = numel(dcms);
intDCM = cell(n, 1);
y = cell(n, 1);

for i = 1:n
    [Ep, M, U]= prepare_interface(dcms{i});    
    intDCM{i} = prepare_data_structure(Ep, M, U);
end

r = dcm_euler_integration(intDCM);

for i = 1:n
    y{i} = forward_model(r{i}.x, r{i}.s, r{i}.f1, r{i}.v1, r{i}.q1, intDCM{i});
end

end


function [intDCM] = prepare_data_structure(Ep, M, U)

if ~isstruct(U), u.u = U; U = u; end
try, dt = U.dt; catch, U.dt = 1; end
 
% number of times to sample (v) and number of microtime bins (u)
%--------------------------------------------------------------------------
u       = size(U.u,1);
try,  v = M.ns;  catch, v = u;   end

DCM.M = M;
DCM.U = U;

Indices = ceil([0:v - 1]*u/v ) + DCM.M.delays(1)/U.dt;

Ep.A = full(Ep.A);
Ep.B = full(Ep.B);
Ep.C = full(Ep.C);
Ep.transit = full(Ep.transit);
Ep.decay = full(Ep.decay);
Ep.epsilon = full(Ep.epsilon);

nr = size(Ep.A,1);

if ~isfield('DCM','delay')
    DCM.delay = ones(nr,1);
end

A = Ep.A';
DCM.b = Ep.B~=0;
mArrayB = permute(Ep.B.*DCM.b, [2 1 3]); % Transpose each nrxnr matrix

if (sum(Ep.D(:))==0)
    Ep.D = zeros(nr,nr,0);
end
DCM.d = Ep.D~=0;
mArrayD = permute(Ep.D.*DCM.d, [2 1 3]); % Transpose each nrxnr matrix
if (isempty(mArrayD))
    mArrayD = zeros(nr,nr,nr);
end


C = DCM.U.u*Ep.C'/16;

U = full(DCM.U.u);

H = [0.64 0.32 2.00 0.32 0.32];

oxygenExtractionFraction = 0.32*ones(nr, 1);
alphainv = 1/H(4);
tau = full(H(3)*exp(Ep.transit));
gamma = H(2);
kappa = full(H(1)*exp(Ep.decay));
epsilon = full(1*exp(Ep.epsilon));

paramList = [DCM.U.dt size(DCM.U.u,1) nr size(DCM.U.u,2) 0 1 1];

switch tapas_mpdcm_compflag
case 0
    intDCM = struct('A', single(A), 'B', single(mArrayB), 'C', single(C), ... 
    'D', single(mArrayD), 'U', single(U), 'rho', ...
    single(oxygenExtractionFraction), 'alphainv', single(alphainv), ...
    'tau', single(tau), 'gamma', single(gamma), 'kappa', single(kappa), ...
    'param', single(paramList));
case 1
     intDCM = struct('A', A, 'C', C, 'D', mArrayD, 'U', U, 'B', mArrayB, ... 
        'rho', oxygenExtractionFraction, 'alphainv', alphainv, 'tau', tau, ...
        'gamma', gamma, 'kappa', kappa, 'param', paramList);
end

intDCM.indices = Indices;
intDCM.epsilon = epsilon;

end


function [y] = forward_model(x, s, f, v, q, intDCM )
% generate the responses per time point

epsilon = intDCM.epsilon;
Indices = intDCM.indices;

relaxationRateSlope  = 25;
frequencyOffset = 40.3;  
oxygenExtractionFraction = 0.4*  ones(1, intDCM.param(3));
echoTime = 0.04;
restingVenousVolume  = 4;

coefficientK1  = 4.3*frequencyOffset * echoTime * oxygenExtractionFraction;
coefficientK2  = epsilon .* (relaxationRateSlope * ...
    oxygenExtractionFraction * echoTime);
coefficientK3  = 1 - epsilon;

if Indices(1) == 0
    Indices(1) = 1;
end

y = restingVenousVolume*( bsxfun(@times,coefficientK1,(1 - (q(Indices,:)))) +...
    bsxfun(@times,coefficientK2,(1 - (q(Indices,:)./v(Indices,:)))) +...
    bsxfun(@times,coefficientK3,(1-v(Indices,:))));        


end


function [Ep, M, U] = prepare_interface(syn_model)

% Check parameters and load specified DCM
%--------------------------------------------------------------------------

DCM       = syn_model;
SNR  = 1;


% Unpack
%--------------------------------------------------------------------------
U     = DCM.U;        % inputs
v     = DCM.v;        % number of scans
n     = DCM.n;        % number of regions
m     = size(U.u,2);  % number of inputs


% check whether this is a nonlinear DCM
%--------------------------------------------------------------------------
if ~isfield(DCM,'d') || isempty(DCM.d)
    DCM.d = zeros(n,n,0);
end

% priors
%--------------------------------------------------------------------------
[pE,pC] = spm_dcm_fmri_priors(DCM.a,DCM.b,DCM.c,DCM.d);


% complete model specification
%--------------------------------------------------------------------------

M.x     = sparse(n,5);
M.pE    = pE;
M.pC    = pC;
M.m     = size(U.u,2);
M.n     = size(M.x(:),1);
M.l     = size(M.x,1);
M.N     = 32;
M.dt    = 16/M.N;
M.ns    = v;


% fMRI slice time sampling
%--------------------------------------------------------------------------
try
    M.delays = DCM.delays; 
catch
    M.delays = zeros(M.n, 1);
end
    
try, M.TE     = DCM.TE;     end


Ep = DCM.Ep;


end
