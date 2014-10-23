function test_mpdcm_fmri_int()
% Test the functionalities of cuda dcm

test_mpdcm_fmri_int_memory()
test_mpdcm_fmri_int_correctness();

end


function test_mpdcm_fmri_int_memory()
%% Checks whether there is any segmentation error in a kamikaze way

display('===========================')
display('Testing mpdcm_fmri_int_memory')
display('===========================')

[u0, theta0, ptheta] = standard_values(8, 8);

u = {u0};
theta = {theta0};

% Don't test for correctness yet, only that there is no segmentation fault 
% while preparing the data.
try
    y = mpdcm_fmri_int(u, theta, ptheta);
    display('    Passed')
catch err
    d = dbstack();
    fprintf('   Not passed at line %d\n', d(1).line)
end

try
    u = cell(2, 1);
    u(:) =  {u0};
    theta = cell(2, 1);
    theta(:) = {theta0};
    y = mpdcm_fmri_int(u, theta, ptheta);
    display('    Passed')
catch err
    d = dbstack();
    fprintf('   Not passed at line %d\n', d(1).line)
end

try
    u = cell(2, 1);
    u(:) =  {u0};
    theta = cell(2, 8);
    theta(:) = {theta0};
    y = mpdcm_fmri_int(u, theta, ptheta);
    display('    Passed')
catch err
    d = dbstack();
    fprintf('   Not passed at line %d\n', d(1).line)
end

try
    u = cell(2, 1);
    u(:) =  {u0};
    theta = cell(2, 1);
    theta(:) = {theta0};
    y = mpdcm_fmri_int(u, theta, ptheta);
    display('    Passed')
catch err
    d = dbstack();
    fprintf('   Not passed at line %d\n', d(1).line)
end

% Test the flags

try
    u = cell(2, 1);
    u(:) =  {u0};
    theta = cell(2, 1);
    theta0.fC = 0;
    theta(:) = {theta0};
    y = mpdcm_fmri_int(u, theta, ptheta);
    display('    Passed')
    theta0.fC = 1;
catch err
    d = dbstack();
    fprintf('   Not passed at line %d\n', d(1).line)
end

% Test different dimensionality for u!

[u0, theta0, ptheta] = standard_values(8, 3);

try
    u = cell(2, 1);
    u(:) =  {u0};
    theta = cell(2, 1);
    theta(:) = {theta0};
    y = mpdcm_fmri_int(u, theta, ptheta);
    display('    Passed')
catch err
    d = dbstack();
    fprintf('   Not passed at line %d\n', d(1).line)
end

dim_u = 8;
dim_x = 8;

u0 = zeros(dim_u, 600);

u0(:, 1) = 20;
u0(:, 90) = 20;
u0(:, 180) = 20;
u0(:, 270) = 20;
u0(:, 360) = 20;
u0(:, 450) = 20;
u0(:, 540) = 20;

% Relaxation rate slope
rrs = 25;
% Oxigen extraction fraction E0
oef = 0.4;
% echo time
et = 0.04;  
% frequency offset of magnetized vessels
fomv = 40.3; 
% Venous volume fraction V0
vvf = 4.0;
% rho
rho = 4.3;

alpha = 0.32;
epsilon = 0.64;
gamma = 0.32;
K = 2.0;
tau = 1.0;

theta0 = struct('A', [], 'B', [], 'C', [], 'epsilon', [], ...
    'K', [], 'tau',  [], 'V0', 1.0, 'E0', 1.0, 'k1', 1.0, 'k2', 1.0, ...
    'k3', 1.0, 'alpha', 1.0, 'gamma', 1.0, 'dim_x', dim_x, 'dim_u', dim_u, ...
    'fA', 1, 'fB', 1, 'fC', 1 );

%%

theta0.A = -0.3*eye(dim_x, dim_x);
B = cell(dim_u, 1);
B(:) = {zeros(dim_x, dim_x)};
theta0.B = B;
theta0.C = eye(dim_x, dim_u);

%%

theta0.epsilon = epsilon*ones(dim_x, 1);
theta0.K = K*ones(dim_x, 1);
theta0.tau = tau*ones(dim_x, 1);

%%

theta0.alpha = alpha;
theta0.gamma = gamma;
theta0.E0 = oef;
theta0.V0 = vvf;
theta0.k1 = rho*fomv*et*oef;
theta0.k2 = epsilon*rrs*oef*et;
theta0.k3 = 1 - epsilon;

theta = {theta0};

ptheta = struct('dt', 1.0, 'dyu', 0.125);

try
    u = cell(2, 1);
    u(:) =  {u0};
    theta = cell(2, 1);
    theta(:) = {theta0};
    y = mpdcm_fmri_int(u, theta, ptheta);
    display('    Passed')
catch err
    d = dbstack();
    fprintf('   Not passed at line %d\n', d(1).line)
end

end

function test_mpdcm_fmri_int_correctness()
%% Test the correctness of the implementation against spm

theta0 = struct('A', [], 'B', [], 'C', [], ...
    'epsilon', [], 'K', [], 'tau',  [], ...
    'V0', [], 'E0', [], 'k1', [], 'k2', [], 'k3', [], ... 
    'alpha', [], 'gamma', [], 'dim_x', [], 'dim_u', [], ...
    'fA', 1, 'fB', 1, 'fC', 1 );

ptheta = struct('dt', 1.0, 'dyu', []);

% Parametrization from spm8

P.decay = 0;
P.transit = 0;
ep = 1;

kappa = 0.64*exp(P.decay);
gamma = 0.32;
alpha = 0.32;
tau = 2*exp(P.transit);
E0 = 0.32;
V0 = 4.0;

r0 = 25;
nu0 = 40.3;
TE = 0.04;


k1 = 4.3*nu0*0.4*TE;
k2 = ep*r0*0.4*TE;
k3 = 1 - ep;

d =  test_mpdcm_fmri_load_td();

theta = cell(5, 1);
u = cell(5, 1);

for i = 1:5

    y = d{i}.y';
    u0 = d{i}.U.u';

    theta0.dim_x = size(y, 1);
    theta0.dim_u = size(u0, 1);

    theta0.A = d{i}.A;
    theta0.fA = 1;

    theta0.B = cell(theta0.dim_u, 1);
    theta0.B(:) = {d{i}.B(:,:,1) d{i}.B(:, :, 2)};
    theta0.fB = 1;
    
    theta0.C = d{i}.C/16;
    theta0.fC = 1;

    theta0.epsilon = ep*ones(theta0.dim_x, 1);
    theta0.K = kappa*ones(theta0.dim_x, 1);
    theta0.tau = tau*ones(theta0.dim_x, 1);
    theta0.E0 = E0;
    theta0.V0 = V0;
    theta0.alpha = alpha;
    theta0.gamma = gamma;
    theta0.k1 = k1;
    theta0.k2 = k2;
    theta0.k3 = k3;

    theta{i} = theta0;
    u{i} = u0;    

end

ptheta.dyu = 2.0*size(y, 2) / size(u0, 2);

ptheta1.dyu = 2.0*size(y, 2) / size(u0, 2);
ptheta1.dt = 0.05;

tic
y0 = mpdcm_fmri_int(u, theta, ptheta);
toc
tic
y1 = mpdcm_fmri_int(u, theta, ptheta1);
toc 

for i = 1:5

    y = d{i}.y';
    u0 = d{i}.U.u';

    theta0.dim_x = size(y, 1);
    theta0.dim_u = size(u0, 1);

    theta0.A = d{i}.A;
    theta0.fA = 1;

    theta0.B = cell(theta0.dim_u, 1);
    theta0.B(:) = {d{i}.B(:,:,1) d{i}.B(:, :, 2)};
    theta0.fB = 1;
    
    theta0.C = d{i}.C/16;
    theta0.fC = 1;

    theta0.epsilon = ep*ones(theta0.dim_x, 1);
    theta0.K = kappa*ones(theta0.dim_x, 1);
    theta0.tau = tau*ones(theta0.dim_x, 1);
    theta0.E0 = E0;
    theta0.V0 = V0;
    theta0.alpha = alpha;
    theta0.gamma = gamma;
    theta0.k1 = k1;
    theta0.k2 = k2;
    theta0.k3 = k3;

    ptheta.dyu = 2.0*size(y, 2) / size(u0, 2);
    
    d{i}.IS = 'spm_int_E';
    d{i}.M.IS = 'spm_int_E';
    
    %tic;
    %cy = spm_dcm_generate(d{i});
    %toc;
    tic;
    theta = cell(20, 1);
    theta(:) = {theta0};
    u = cell(20, 1);
    %u(:) = {u0};
    u(:) = {u0
    ny = mpdcm_fmri_int(u, theta, ptheta);
    toc;

    if all(abs(d{i}.y - ny{1}) < 1e-2)
        display('    Passed')
    else
        td = dbstack();
        fprintf('   Not passed at line %d\n', td(1).line)
        figure(); 
        hold on; 
        plot(ny{1}); 
        plot(d{i}.y, '.'); 
        %plot(cy.y, 'k');
    end

end


end

function [u, theta, ptheta] = standard_values(dim_x, dim_u)
%% Returns a parametrization that is expected to work properly

u = zeros(dim_u, 600);

u(:, 1) = 20;
u(:, 90) = 20;
u(:, 180) = 20;
u(:, 270) = 20;
u(:, 360) = 20;
u(:, 450) = 20;
u(:, 540) = 20;

theta = struct('A', [], 'B', [], 'C', [], 'epsilon', [], ...
    'K', [], 'tau',  [], 'V0', 1.0, 'E0', 1.0, 'k1', 1.0, 'k2', 1.0, ...
    'fA', 1, 'fB', 1, 'fC', 1, ...
    'k3', 1.0, 'alpha', 1.0, 'gamma', 1.0, 'dim_x', dim_x, 'dim_u', dim_u);

theta.A = -0.3*eye(dim_x);

B = cell(dim_u, 1);
B(:) = {zeros(dim_x, dim_x)};
theta.B = B;

theta.C = zeros(dim_x, dim_u);

theta.epsilon = zeros(dim_x, 1);
theta.K = zeros(dim_x, 1);
theta.tau = zeros(dim_x, 1);

ptheta = struct('dt', 1.0, 'dyu', 0.125);

end
