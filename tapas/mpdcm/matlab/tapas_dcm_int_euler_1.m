function y = tapas_dcm_int_euler_1(Ep, M, U)

if ~isstruct(U), u.u = U; U = u; end
try, dt = U.dt; catch, U.dt = 1; end
 
% number of times to sample (v) and number of microtime bins (u)
%--------------------------------------------------------------------------
u       = size(U.u,1);
try,  v = M.ns;  catch, v = u;   end


%ceil([(1:v) - 0.5]*u/v);


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

% Transform to single if necessary
switch tapas_mpdcm_compflag
case 0
    [x,s,f,v,q]  = dcm_euler_integration(single(A), single(C), single(U), ...
        single(mArrayB), single(mArrayD), single(oxygenExtractionFraction), ...
        single(alphainv), single(tau), single(gamma), single(kappa), ...
        single(paramList));
case 1
    [x,s,f,v,q]  = dcm_euler_integration(A,C,U,mArrayB,mArrayD,...
        oxygenExtractionFraction,alphainv,tau,gamma,kappa,paramList); 
end       
               
               
% generate the responses per time point
relaxationRateSlope  = 25;
frequencyOffset = 40.3;  
oxygenExtractionFraction = 0.4*ones(1,paramList(3));
echoTime = 0.04;
restingVenousVolume  = 4;

coefficientK1  = 4.3*frequencyOffset*echoTime*oxygenExtractionFraction;
coefficientK2  = epsilon.*(relaxationRateSlope*oxygenExtractionFraction*echoTime);
coefficientK3  = 1 - epsilon;

if Indices(1) == 0
    Indices(1) = 1;
end

y = restingVenousVolume*( bsxfun(@times,coefficientK1,(1 - (q(Indices,:)))) +...
            bsxfun(@times,coefficientK2,(1 - (q(Indices,:)./v(Indices,:)))) +...
            bsxfun(@times,coefficientK3,(1-v(Indices,:))));        

end

