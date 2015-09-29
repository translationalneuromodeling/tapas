function test_tapas_mpdcm_fmri_int_check_input(fp)
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

[u0, theta0, ptheta] = standard_values(8, 8);
u = {u0};
theta = {theta0};

try
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    display('   Passed');   
catch err
    db = dbstack();
    fprintf(fp, '   Not passed at line %d\n', db(1).line)
    disp(getReport(err, 'extended'));
end

% Test u

try
    u = zeros(10, 10);
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, 'mpdcm:fmri:int:input:u:not_cell')
        display('   Passed')
    else
        db = dbstack();
        fprintf(fp, '   Not passed at line %d\n', db(1).line)
        disp(getReport(err, 'extended'));
    end
end
try
    u = cell(3, 3, 3);
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, 'mpdcm:fmri:int:input:u:ndim')
        display('   Passed')
    else
        db = dbstack();
        fprintf(fp, '   Not passed at line %d\n', db(1).line)
        disp(getReport(err, 'extended'));
    end
end
try
    u = cell(2, 2);
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, 'mpdcm:fmri:int:input:u:dsize')
        display('   Passed')
    else
        db = dbstack();
        fprintf(fp, '   Not passed at line %d\n', db(1).line)
        disp(getReport(err, 'extended'));
    end
end
try
    u = {{}};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, 'mpdcm:fmri:int:input:u:cell:not_numeric')
        display('   Passed')
    else
        db = dbstack();
        fprintf(fp, '   Not passed at line %d\n', db(1).line)
        disp(getReport(err, 'extended'));
    end
end
try
    u = {j*ones(3, 3)};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, 'mpdcm:fmri:int:input:u:cell:not_real')
        display('   Passed')
    else
        db = dbstack();
        fprintf(fp, '   Not passed at line %d\n', db(1).line)
        disp(getReport(err, 'extended'));
    end
end
try
    u = {sparse(eye(5))};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, 'mpdcm:fmri:int:input:u:cell:sparse')
        display('   Passed')
    else
        db = dbstack();
        fprintf(fp, '   Not passed at line %d\n', db(1).line)
        disp(getReport(err, 'extended'));
    end
end

try
    u = cell(2, 1);
    u{1} = zeros(4, 50);
    u{2} = zeros(5, 50);
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, 'mpdcm:fmri:int:input:u:cell:not_match')
        display('   Passed')
    else
        db = dbstack();
        fprintf(fp, '   Not passed at line %d\n', db(1).line)
        disp(getReport(err, 'extended'));
    end
end

% Test theta

u = {u0};

try
    theta = zeros(3, 3);
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, 'mpdcm:fmri:int:input:theta:not_cell')
        display('   Passed')
    else
        db = dbstack();
        fprintf(fp, '   Not passed at line %d\n', db(1).line)
        disp(getReport(err, 'extended'));
    end
end

try
    theta = cell(3, 3, 3);
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, 'mpdcm:fmri:int:input:theta:ndim')
        display('   Passed')
    else
        db = dbstack();
        fprintf(fp, '   Not passed at line %d\n', db(1).line)
        disp(getReport(err, 'extended'));
    end
end

try
    theta = {zeros(3, 3)};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, 'mpdcm:fmri:int:input:theta:cell:not_struct')
        display('   Passed')
    else
        db = dbstack();
        fprintf(fp, '   Not passed at line %d\n', db(1).line)
        disp(getReport(err, 'extended'));
    end
end

try
    theta = {struct('A', {1, 1})};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, ...
        'mpdcm:fmri:int:input:theta:cell:ndim')
        display('   Passed')
    else
        db = dbstack();
        fprintf(fp, '   Not passed at line %d\n', db(1).line)
        disp(getReport(err, 'extended'));
    end
end


try
    theta = {struct()};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, ...
        'mpdcm:fmri:int:input:theta:cell:dim_x:missing')
        display('   Passed')
    else
        db = dbstack();
        fprintf(fp, '   Not passed at line %d\n', db(1).line)
        disp(getReport(err, 'extended'));
    end
end

try
    theta = {struct('dim_x', zeros(3, 3))};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, ...
        'mpdcm:fmri:int:input:theta:cell:dim_x:dsize')
        display('   Passed')
    else
        db = dbstack();
        fprintf(fp, '   Not passed at line %d\n', db(1).line)
        disp(getReport(err, 'extended'));
    end
end
try
    theta = {struct('dim_x', 8)};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, ...
        'mpdcm:fmri:int:input:theta:cell:dim_u:missing')
        display('   Passed')
    else
        db = dbstack();
        fprintf(fp, '   Not passed at line %d\n', db(1).line)
        disp(getReport(err, 'extended'));
    end
end
try
    theta = {struct('dim_x', 8, 'dim_u', [])};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, ...
        'mpdcm:fmri:int:input:theta:cell:dim_u:dsize')
        display('   Passed')
    else
        db = dbstack();
        fprintf(fp, '   Not passed at line %d\n', db(1).line)
        disp(getReport(err, 'extended'));
    end
end
try
    theta = {struct('dim_x', 8, 'dim_u', 8)};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, ...
        'mpdcm:fmri:int:input:theta:cell:fA:missing')
        display('   Passed')
    else
        d = dbstack();
        fprintf('   Not passed at line %d\n', d(1).line)
    end
end

% Should work fine for all the flags

try
    theta = {struct('dim_x', 8, 'dim_u', 8, 'fA', 1, 'fB', 1, 'fC', 1, ...
        'fD', 0)};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, ...
        'mpdcm:fmri:int:input:theta:cell:A:missing')
        display('   Passed')
    else
        d = dbstack();
        fprintf('   Not passed at line %d\n', d(1).line)
    end
end

try
    theta = {struct('dim_x', 8, 'dim_u', 8, 'fA', 1, 'fB', 1, 'fC', 1, ...
        'fD', 0)};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, ...
        'mpdcm:fmri:int:input:theta:cell:A:missing')
        display('   Passed')
    else
        d = dbstack();
        fprintf('   Not passed at line %d\n', d(1).line)
    end
end

try
    theta = {struct('dim_x', 8, 'dim_u', 8, 'fA', 1, 'fB', 1, 'fC', 1, ...
        'fD', 0, 'A', struct('A', 0))};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, ...
        'mpdcm:fmri:int:input:theta:cell:A:not_numeric')
        display('   Passed')
    else
        d = dbstack();
        fprintf('   Not passed at line %d\n', d(1).line)
    end
end

try
    theta = {struct('dim_x', 8, 'dim_u', 8, 'fA', 1, 'fB', 1, 'fC', 1, ...
        'fD', 0, 'A', struct('A', 0))};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, ...
        'mpdcm:fmri:int:input:theta:cell:A:not_numeric')
        display('   Passed')
    else
        d = dbstack();
        fprintf('   Not passed at line %d\n', d(1).line)
    end
end

try
    theta = {struct('dim_x', 8, 'dim_u', 8, 'fA', 1, 'fB', 1, 'fC', 1, ...
        'fD', 0, 'A', zeros(3, 3, 3))};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, ...
        'mpdcm:fmri:int:input:theta:cell:A:ndim')
        display('   Passed')
    else
        d = dbstack();
        fprintf('   Not passed at line %d\n', d(1).line)
    end
end

try
    theta = {struct('dim_x', 8, 'dim_u', 8, 'fA', 1, 'fB', 1, 'fC', 1, ...
        'fD', 0, 'A', j*eye(3))};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, ...
        'mpdcm:fmri:int:input:theta:cell:A:not_real')
        display('   Passed')
    else
        d = dbstack();
        fprintf('   Not passed at line %d\n', d(1).line)
    end
end

try
    theta = {struct('dim_x', 8, 'dim_u', 8, 'fA', 1, 'fB', 1, 'fC', 1, ...
        'fD', 0, 'A', sparse(eye(3)))};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, ...
        'mpdcm:fmri:int:input:theta:cell:A:sparse')
        display('   Passed')
    else
        d = dbstack();
        fprintf('   Not passed at line %d\n', d(1).line)
    end
end

try
    theta = {struct('dim_x', 8, 'dim_u', 8, 'fA', 1, 'fB', 1, 'fC', 1, ...
        'fD', 0, 'A', zeros(3, 3, 3))};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, ...
        'mpdcm:fmri:int:input:theta:cell:A:ndim')
        display('   Passed')
    else
        d = dbstack();
        fprintf('   Not passed at line %d\n', d(1).line)
    end
end

try
    theta = {struct('dim_x', 8, 'dim_u', 8, 'fA', 1, 'fB', 1, 'fC', 1, ...
        'fD', 0, 'A', zeros(7, 7))};
    tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);
    error('    Not passed')
catch err
    if strcmp(err.identifier, ...
        'mpdcm:fmri:int:input:theta:cell:A:dsize')
        display('   Passed')
    else
        d = dbstack();
        fprintf('   Not passed at line %d\n', d(1).line)
    end
end

% TODO test B, C and D ...


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
B = zeros(dim_x, dim_x, dim_u);
theta.B = B;

theta.C = zeros(dim_x, dim_u);

theta.epsilon = zeros(dim_x, 1);
theta.K = zeros(dim_x, 1);
theta.tau = zeros(dim_x, 1);

ptheta = struct('dt', 1.0, 'dyu', 0.125);

end
