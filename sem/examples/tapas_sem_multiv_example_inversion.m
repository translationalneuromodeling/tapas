function tapas_sem_multiv_example_inversion(model, param, fp)
%% Test 
%
% fp -- Pointer to a file for the test output, defaults to 1
%
% aponteeduardo@gmail.com
% copyright (C) 2015
%

n = 0,

n = n + 1;
if nargin < n
    model = 'seri';
end

n = n + 1;
if nargin < n
    param = 'invgamma';
end

n = n + 1;
if nargin < n
    fp = 1;
end

fname = mfilename();
fname = regexprep(fname, 'test_', '');


fprintf(fp, '================\n Test %s\n================\n', fname);

[data] = prepare_data();

switch model
case 'seri'
    ptheta = tapas_sem_seri_invgamma_ptheta(); % Choose at convinience.
    switch param
    case 'invgamma'
        ptheta.llh = @c_seri_multi_invgamma;
    case 'gamma'
        ptheta.llh = @c_seri_multi_gamma;
    case 'mixedgamma'
        ptheta.llh = @c_seri_multi_mixedgamma;
    case 'lognorm'
        ptheta.llh = @c_seri_multi_lognorm;
    case 'later'
        ptheta.llh = @c_seri_multi_later;
    case 'wald'
        ptheta.llh = @c_seri_multi_wald;
    end

    ptheta.jm = [...
        eye(19)
        zeros(3, 8) eye(3) zeros(3, 8)];

    ptheta.x = ones(4, 1);

    pars = struct();


case 'dora'
    ptheta = tapas_sem_dora_invgamma_ptheta(); 
    switch param
    case 'invgamma'
        ptheta.llh = @c_dora_multi_invgamma;
    case 'gamma'
        ptheta.llh = @c_dora_multi_gamma;
    case 'mixedgamma'
        ptheta.llh = @c_dora_multi_mixedgamma;
    case 'lognorm'
        ptheta.llh = @c_dora_multi_lognorm;
    case 'later'
        ptheta.llh = @c_dora_multi_later;
    case 'wald'
        ptheta.llh = @c_dora_multi_wald;
    end

    ptheta.jm = [...
        eye(19)
        zeros(3, 8) eye(3) zeros(3, 8)];

    ptheta.x = ones(4, 1);

case 'prosa'
    ptheta = tapas_sem_prosa_invgamma_ptheta(); % Choose at convinience.
    switch param
    case 'invgamma'
        ptheta.llh = @c_prosa_multi_invgamma;
    case 'gamma'
        ptheta.llh = @c_prosa_multi_gamma;
    case 'mixedgamma'
        ptheta.llh = @c_prosa_multi_mixedgamma;
    case 'lognorm'
        ptheta.llh = @c_prosa_multi_lognorm;
    case 'later'
        ptheta.llh = @c_prosa_multi_later;
    case 'wald'
        ptheta.llh = @c_prosa_multi_wald;
    end

    ptheta.jm = [...
        eye(15)
        zeros(3, 6) eye(3) zeros(3, 6)];

    ptheta.x = ones(4, 1);

end

pars = struct();

pars.T = ones(4, 1) * linspace(0.1, 1, 8).^5;
pars.nburnin = 4000;
pars.niter = 4000;
pars.ndiag = 1000;
pars.mc3it = 16;
pars.verbose = 1;

inference = struct();
tic
tapas_sem_multiv_estimate(data, ptheta, inference, pars);
toc


end

function [data] = prepare_data()
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

data = struct('y', cell(3, 1), 'u', []);
j = 1;
for i = unique(u.b)'
    data(j).y = struct();
    data(j).u = struct();

    data(j).y.a = y.a(u.b == i);
    data(j).y.t = y.t(u.b == i);
    data(j).y.i = y.i(u.b == i);

    data(j).u.s = u.s(u.b == i);
    data(j).u.b = u.b(u.b == i);
    data(j).u.tt = u.tt(u.b == i);


    j = j + 1;
end

end
