function tapas_sem_example_inversion(model, fp)
%% Test 
%
% fp -- Pointer to a file for the test output, defaults to 1
%
% aponteeduardo@gmail.com
% copyright (C) 2015
%

n = 0;

n = n + 1;
if nargin < n
    model = 2;
end

n = n + 1;
if nargin < n
    fp = 1;
end

fname = mfilename();
fname = regexprep(fname, 'test_', '');

fprintf(fp, '================\n Test %s\n================\n', fname);

[y, u] = prepare_data();

if model == 1

ptheta = tapas_sem_prosa_invgamma_ptheta(); % Choose at convinience.
htheta = tapas_sem_prosa_htheta(); % Choose at convinience.
ptheta.ptrans = @(vtheta) vtheta;

% Insert a parametrization matrix

% Assume all the delays are equal but otherwise have free parameters

% 12 unit parameters and 2 delays and rate of outliers.
ptheta.jm = [eye(15) 
    zeros(3, 6) eye(3) zeros(3, 6)]; % Share the parameters across trial types

pars = struct();

pars.T = linspace(0.1, 1, 16).^5;
pars.nburnin = 3000;
pars.niter = 3000;
pars.kup = 1000;
pars.mc3it = 16;
pars.verbose = 1;

tapas_sem_estimate(y, u, ptheta, htheta, pars);

end

if model == 2

fprintf(1, 'Seri inversion\n')

ptheta = tapas_sem_seri_wald_ptheta(); % Choose at convinience.
htheta = tapas_sem_seri_htheta(); % Choose at convinience.

% Insert a parametrization matrix

% The same parameters are used in pro and antisaccade trials
ptheta.jm = [...
    eye(19)
    zeros(3, 8) eye(3) zeros(3, 8)];

pars = struct();

pars.T = linspace(0.1, 1, 16).^5;
pars.nburnin = 2000;
pars.niter = 2000;
pars.kup = 1000;
pars.mc3it = 16;
pars.verbose = 1;

tapas_sem_estimate(y, u, ptheta, htheta, pars);

end

if model == 3

fprintf(1, 'Dora inversion\n');

ptheta = tapas_sem_dora_invgamma_ptheta(); 
htheta = tapas_sem_dora_htheta(); % Choose at convinience.

% Insert a parametrization matrix

% The same parameters are used in pro and antisaccade trials
ptheta.jm = [...
    eye(19)
    zeros(3, 8) eye(3) zeros(3, 8)];

pars = struct();

pars.T = linspace(0.1, 1, 16).^5;
pars.nburnin = 3000;
pars.niter = 3000;
pars.kup = 1000;
pars.mc3it = 16;
pars.verbose = 1;

tapas_sem_estimate(y, u, ptheta, htheta, pars);

end


end

function [y, u] = prepare_data()
%% Prepares the test data

NDTIME = 100;

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
                      

y.a = y.a(~y.i);
y.t = y.t(~y.i);

u.s = u.s(~y.i);
u.b = u.b(~y.i);
u.tt = u.tt(~y.i);

y.i = y.i(~y.i);
end
