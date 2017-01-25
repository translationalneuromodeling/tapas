function tapas_sem_example_inversion(fp)
%% Test 
%
% fp -- Pointer to a file for the test output, defaults to 1
%
% aponteeduardo@gmail.com
% copyright (C) 2015
%

if nargin < 1
    fp = 1;
end

fname = mfilename();
fname = regexprep(fname, 'test_', '');


fprintf(fp, '================\n Test %s\n================\n', fname);

[y, u] = prepare_data();

ptheta = sooner_ware_ptheta(); % Choose at convinience.
htheta = sooner_ware_htheta(); % Choose at convinience.

pars = struct();

pars.T = linspace(0.0001, 1, 10).^5;
pars.nburnin = 1000;
pars.niter = 500;
pars.mc3 = 0;
pars.verbose = 1;

% Test whether there is any clear bug
try
    tic
    sooner_estimate(y, u, ptheta, htheta, pars);
    toc
    fprintf(fp, '       Passed\n');
catch err
    fprintf(fp, '   Not passed at line %d\n', err.stack(end).line);
    fprintf(fp, getReport(err, 'extended'));
end


end

function [y, u] = prepare_data()
%% Prepares the test data

NDTIME = 120;

f = mfilename('fullpath');
[tdir, ~, ~] = fileparts(f);

% Files are delimited with a tab and skip the header
d = dlmread(fullfile(tdir, 'data', 'sbj02.csv'), '\t', 1, 0);

%Filter out unreasonably short reactions

nt = size(d, 1);

y = struct('t', [], 'a', [], 'b', []);

% Subject and block
u.s = d(:, 1);
u.b = d(:, 2);

% Invalid trials are shorter than 100 ms
y.i = d(:, 7) < NDTIME;
% Shift and rescale the data
y.t = d(:, 7)/100;

% Is it a prosaccade or an antisaccade
lr = zeros(nt, 1);
% Saccade to the left
lr(d(:, 6) < 640) = 1;
% Up to hear prosaccades are 1 and antisaccades are 0
y.a = lr == d(:, 5);
y.a = double(y.a);

u.tt = d(:, 4);

% Matlab and python conventions don't aggree

t0 = y.a == 0;
t1 = y.a == 1;

y.a(t0) = 1;
y.a(t1) = 0;

t0 = u.tt == 0;
t1 = u.tt == 1;
                      

end
