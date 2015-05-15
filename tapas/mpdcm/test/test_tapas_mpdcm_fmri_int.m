function test_tapas_mpdcm_fmri_int(fp)
%% Test 
%
% fp -- Pointer to a file for the test output, defaults to 1
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

if nargin < 1
    fp = 1;
end

fname = mfilename();
fname = regexprep(fname, 'test_', '');


fprintf(fp, '================\n Test %s\n================\n', fname);

test_tapas_mpdcm_fmri_int_memory(fp, 'euler')
test_tapas_mpdcm_fmri_int_correctness(fp, 'euler');

test_tapas_mpdcm_fmri_int_memory(fp, 'kr4');
test_tapas_mpdcm_fmri_int_correctness(fp, 'kr4');

test_tapas_mpdcm_fmri_int_memory(fp, 'bs');
test_tapas_mpdcm_fmri_int_correctness(fp, 'bs');

end

function test_tapas_mpdcm_fmri_int_memory(fp, integ)
%% Checks whether there is any segmentation error in a kamikaze way

[u0, theta0, ptheta] = standard_values(8, 8);

ptheta.integ = integ;

u = {u0};
theta = {theta0};

% Don't test for correctness yet, only that there is no segmentation fault 
% while preparing the data.
try
    y = tapas_mpdcm_fmri_int(u, theta, ptheta);
    fprintf(fp, '    Passed\n')
catch err
    db = dbstack();
    fprintf('   Not passed at line %d\n', db(1).line)
    disp(getReport(err, 'extended'));
end

try
    u = cell(2, 1);
    u(:) =  {u0};
    theta = cell(2, 1);
    theta(:) = {theta0};
    y = tapas_mpdcm_fmri_int(u, theta, ptheta);
    fprintf(fp, '    Passed\n')
catch err
    fprintf(fp, '   Not passed at line %d\n', err.stack(end).line);
    fprintf(fp, getReport(err, 'extended'));
end

try
    u = cell(2, 1);
    u(:) =  {u0};
    theta = cell(2, 8);
    theta(:) = {theta0};
    y = tapas_mpdcm_fmri_int(u, theta, ptheta);
    fprintf(fp, '    Passed\n')
catch err
    fprintf(fp, '   Not passed at line %d\n', err.stack(end).line);
    fprintf(fp, getReport(err, 'extended'));
end

try
    u = cell(2, 1);
    u(:) =  {u0};
    theta = cell(2, 1);
    theta(:) = {theta0};
    y = tapas_mpdcm_fmri_int(u, theta, ptheta);
    fprintf(fp, '    Passed\n')
catch err
    d = dbstack();
    fprintf(fp, '   Not passed at line %d\n', d(1).line)
end

% Test the flags

try
    u = cell(2, 1);
    u(:) =  {u0};
    theta = cell(2, 1);
    theta0.fC = 0;
    theta(:) = {theta0};
    y = tapas_mpdcm_fmri_int(u, theta, ptheta);
    fprintf(fp, '    Passed\n')
    theta0.fC = 1;
catch err
    d = dbstack();
    fprintf(fp, '   Not passed at line %d\n', d(1).line)
end

% Test different dimensionality for u!

[u0, theta0, ptheta] = standard_values(8, 3);

ptheta.integ = integ;

try
    u = cell(2, 1);
    u(:) =  {u0};
    theta = cell(2, 1);
    theta(:) = {theta0};
    y = tapas_mpdcm_fmri_int(u, theta, ptheta);
    fprintf(fp, '    Passed\n')
catch err
    d = dbstack();
    fprintf(fp, '   Not passed at line %d\n', d(1).line)
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
    'fA', 1, 'fB', 1, 'fC', 1 , 'fD', 0);

%%

theta0.A = -0.3*eye(dim_x, dim_x);
theta0.B = zeros(dim_x, dim_x, dim_u);
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

ptheta = struct('dt', 1.0, 'dyu', 0.125, 'integ', integ);

try
    u = cell(2, 1);
    u(:) =  {u0};
    theta = cell(2, 1);
    theta(:) = {theta0};
    y = tapas_mpdcm_fmri_int(u, theta, ptheta);
    fprintf(fp, '    Passed\n')
catch err
    d = dbstack();
    fprintf(fp, '   Not passed at line %d\n', d(1).line)
end

end

function test_tapas_mpdcm_fmri_int_correctness(fp, integ)
%% Test the correctness of the implementation against spm

tol = 2e-1;

d =  test_tapas_mpdcm_fmri_load_td();

theta0 = struct('A', [], 'B', [], 'C', [], ...
    'epsilon', [], 'K', [], 'tau',  [], ...
    'V0', [], 'E0', [], 'k1', [], 'k2', [], 'k3', [], ... 
    'alpha', [], 'gamma', [], 'dim_x', [], 'dim_u', [], ...
    'fA', 1, 'fB', 1, 'fC', 1 , 'fD', 0);

ptheta = struct('dt', 1.0, 'dyu', [], 'integ', integ);

% Parametrization from spm8

P.decay = 0;
P.transit = 0;
ep = 1;

kappa = 0.64;
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

theta = cell(5, 1);
u = cell(5, 1);

for i = 1:5

    y = d{i}.y';
    u0 = d{i}.U.u';

    theta0.dim_x = size(y, 1);
    theta0.dim_u = size(u0, 1);

    theta0.A = d{i}.A;
    theta0.fA = 1;

    theta0.B = d{i}.B; 
    theta0.fB = 1;
    
    theta0.C = d{i}.C;
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

dcm = cell(5, 1);
for i = 1:5
    dcm{i} = tapas_mpdcm_spm_dcm_generate(d{i});
end

d = dcm;

for i = 1:5
    dcm = d{i};
    [y, u, theta, ptheta] = tapas_mpdcm_fmri_tinput({dcm});
    ptheta.integ = integ;
    theta{1}.A = dcm.A;
    theta{1}.B = dcm.B;
    theta{1}.C = dcm.C;
    ny = tapas_mpdcm_fmri_int(u, theta, ptheta);
    if all(abs(dcm.y - ny{1}) < tol)
        fprintf(fp, '    Passed\n')
    else
        td = dbstack();
        fprintf(fp, '   Not passed at line %d\n', td(1).line)
    end

end

[y, u, theta, ptheta] = tapas_mpdcm_fmri_tinput(d);

ptheta.integ = integ;

for i = 1:5
    dcm = d{i};
    theta{i}.A = dcm.A;
    theta{i}.B = dcm.B;
    theta{i}.C = dcm.C;
end
ny = tapas_mpdcm_fmri_int(u, theta, ptheta);

for i = 1:5
    if all(abs(d{i}.y - ny{i}) < tol)
        fprintf(fp, '    Passed\n')
    else
        td = dbstack();
        fprintf(fp, '   Not passed at line %d\n', td(1).line)
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
    'K', [], 'tau',  [], 'V0', 1.0, 'E0', 0.7, 'k1', 1.0, 'k2', 1.0, ...
    'fA', 1, 'fB', 1, 'fC', 1, 'fD', 0, ...
    'k3', 1.0, 'alpha', 1.0, 'gamma', 1.0, 'dim_x', dim_x, 'dim_u', dim_u);

theta.A = -0.3*eye(dim_x);
theta.B = zeros(dim_x, dim_x, dim_u);
theta.C = zeros(dim_x, dim_u);

theta.epsilon = ones(dim_x, 1);
theta.K = ones(dim_x, 1);
theta.tau = ones(dim_x, 1);

ptheta = struct('dt', 1.0, 'dyu', 0.125);

end
