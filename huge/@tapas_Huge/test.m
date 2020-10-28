function [ ] = test( )
% Unit testing for methods of class tapas_Huge
%

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2019 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <https://www.gnu.org/licenses/>.
% 
% This software is provided "as is", without warranty of any kind, express
% or implied, including, but not limited to the warranties of
% merchantability, fitness for a particular purpose and non-infringement.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is under active development.
% Considerable changes may occur in future releases. For support please
% refer to:
% https://github.com/translationalneuromodeling/tapas/issues
% 


%% set up DCM network structure
rng('shuffle')

% define DCM network structure
dcm = struct( );
dcm.n = 3;
dcm.a = logical([ ...
    1 0 0; ...
    1 1 1; ...
    1 1 1; 
]);
dcm.c = false(dcm.n, 3);
dcm.c([1, 5]) = true;
dcm.b = false(dcm.n, dcm.n, 3);
dcm.b(:, :, 3) = logical([ ...
    0 0 0; ...
    1 0 1; ...
    1 1 0; ...
]);
dcm.d = false(dcm.n, dcm.n, 0);

% generate experimental stimuli
U = struct();
U.dt = 1.84/16;
tmp = tapas_huge_boxcar(U.dt, [24*13 24], [2 26], [3/4 16/26], [0 0;0 0]);
nSmp = length(tmp{1}) + 160;
tmp{1}(nSmp) = 0;
tmp{2}(nSmp) = 0;
tmp{2} = tmp{1}.*tmp{2};
tmp{3} = zeros(1, 24);
tmp{3}([2 3 1 4 3 1] + (0:5)*4) = 1;
tmp{3} = reshape(repmat(tmp{3}, round(26/U.dt), 1), [], 1);
tmp{3}(nSmp) = 0;
tmp{3} = tmp{3}.*tmp{2};
tmp{4} = zeros(1, 24);
tmp{4}([4 2 3 1 1 4] + (0:5)*4) = 1;
tmp{4} = reshape(repmat(tmp{4}, round(26/U.dt), 1), [], 1);
tmp{4}(nSmp) = 0;
tmp{4} = tmp{4}.*tmp{2};
tmp = tmp(2:4);

U.u = circshift(cell2mat(tmp), 6*16, 1);
U.name = {'G', 'CG', 'SG'};
dcm.U = U;
dcm.Y.dt = 16*dcm.U.dt;
dcm.Y.name = {'LSA'; 'V4'; 'SPL'};
dcm.TE = .03;

sigma = .141; % group standard deviation
listGroups = cell(2, 1);

% group 1
dcm.Ep.A = [-.4  .0  .0; ...
             .2 -.4  .0; ...
             .0  .1 -.4;];
dcm.Ep.B = zeros(dcm.n, dcm.n, 3);
dcm.Ep.B(2, 1, 3) = .2;
dcm.Ep.B(3, 2, 3) = .35;
dcm.Ep.C = double(dcm.c);
dcm.Ep.D = double(dcm.d);
dcm.Ep.transit = zeros(dcm.n,1);
dcm.Ep.decay = zeros(dcm.n,1);
dcm.Ep.epsilon = 0;
tmp = [dcm.a(:);dcm.b(:);dcm.c(:);dcm.d(:)];
dcm.Cp = diag([double(tmp).*sigma.^2; ones(2*dcm.n+1, 1)*exp(-6)]);
listGroups{1} = dcm;

% group 2
dcm.Ep.A = [-.4  .0   .0; ...
             .0 -.4   .1; ...
             .4 -.15 -.4;];
dcm.Ep.B = zeros(dcm.n, dcm.n, 3);
dcm.Ep.B(3, 1, 3) = .35;
dcm.Ep.B(2, 3, 3) = .35;
listGroups{2} = dcm;

%% test: compiling
fprintf('Testing mex compilation: \n')

tapas_huge_compile();

fprintf('passed.\n')

%% test: constructor
fprintf('Testing constructor: ')

obj = tapas_Huge('Tag', 'testing');
assert(obj.K == 1, 'tapas:huge:test', 'constructor default K');
assert(strcmp(obj.tag, 'testing'), 'tapas:huge:test', 'constructor tag');

fprintf('passed.\n')


%% test: simulate
fprintf('Testing data generation: ')

groupSizes = [10 10]; % simulate 10 subjects for each group
snr = 1; % using a signal-to-noise ratio of 1 (0 dB)
obj = obj.simulate(listGroups, groupSizes, 'Snr', snr);

assert(~isempty(obj.model), 'tapas:huge:test', 'simulate model');
assert(obj.K == 2, 'tapas:huge:test', 'simulate K');
assert(obj.N == sum(groupSizes), 'tapas:huge:test', 'simulate N');
assert(obj.N == length(obj.data), 'tapas:huge:test', 'data = N');
assert(obj.N == length(obj.inputs), 'tapas:huge:test', 'inputs = N');

fprintf('passed.\n')

%% test: export
fprintf('Testing data export: ')

[ listDcms, listConfounds ] = obj.export( );
assert(length(listDcms) == sum(groupSizes), 'tapas:huge:test', 'export N');
assert(isempty(listConfounds), 'tapas:huge:test', 'export N');

fprintf('passed.\n')


%% test: import
fprintf('Testing data import: ')

obj = tapas_Huge('Tag', 'import');
obj = obj.import(listDcms);

assert(obj.N == sum(groupSizes), 'tapas:huge:test', 'import N');
assert(obj.N == length(obj.data), 'tapas:huge:test', 'data = N');
assert(obj.N == length(obj.inputs), 'tapas:huge:test', 'inputs = N');
assert(isempty(obj.posterior), 'tapas:huge:test', 'import posterior');

fprintf('passed.\n')


%% test: estimate
fprintf('Testing model inversion: \n')

obj = obj.estimate('K', 1);
assert(obj.K == 1, 'tapas:huge:test', 'estimate K1');
assert(all(size(obj.posterior.q_nk) == [obj.N obj.K]), ...
    'tapas:huge:test', 'estimate q_nk1');
obj = obj.estimate('K', 2);
assert(obj.K == 2, 'tapas:huge:test', 'estimate K2');
assert(all(size(obj.posterior.q_nk) == [obj.N obj.K]), ...
    'tapas:huge:test', 'estimate q_nk2');
obj = obj.estimate('K', 3);
assert(obj.K == 3, 'tapas:huge:test', 'estimate K3');
assert(all(size(obj.posterior.q_nk) == [obj.N obj.K]), ...
    'tapas:huge:test', 'estimate q_nk3');

fprintf('passed.\n')

%% test: optional arguments
fprintf('Testing optional arguments: \n')

obj2 = obj.estimate('K', 2, 'Dcm', listDcms, 'OmitFromClustering', ...
    struct('a',1,'c',1), 'NumberOfIterations', 25);
assert(obj2.K == 2, 'tapas:huge:test', 'nvp K');
assert(all(size(obj2.posterior.q_nk) == [obj2.N obj2.K]), ...
    'tapas:huge:test', 'nvp q_nk2');
s = diag([exp(-6)*ones(1,3),1,1,exp(-6)*ones(1,3*2+1)]);
assert(all(obj2.prior.Sigma_h(:) == s(:)), 'tapas:huge:test', 'nvp Sigma_h');
assert(obj2.options.nvp.numberofiterations == 25, 'tapas:huge:test', ...
    'nvp nIt');

fprintf('passed.\n')

%% test: save
fprintf('Testing saving: ')

save( 'tapas_Huge_unit_test', obj, 'K', 'L', 'N', 'R', 'idx', 'dcm', ... 
    'inputs', 'data', 'labels', 'options', 'prior', 'posterior', ... 
    'trace', 'model' )
load('tapas_Huge_unit_test.mat');
assert(K == obj.K, 'tapas:huge:test', 'save K');
assert(N == obj.N, 'tapas:huge:test', 'save N');
assert(isempty(model), 'tapas:huge:test', 'save model');

fprintf('passed.\n')

%% test: un-vectorize
fprintf('Testing unvec: ')

th = (1:obj.idx.P_c+obj.idx.P_h)';
[A, B, C, D, tau, kappa, epsilon] = obj.theta2abcd(th, obj.idx, obj.R, obj.L);
th2 = [A(:);B(:);C(:);D(:);tau(:);kappa(:);epsilon(:)];
idx = [obj.dcm.a(:);obj.dcm.b(:);obj.dcm.c(:);...
    obj.dcm.d(:);true(obj.idx.P_h,1)];
assert(all(th2(idx) == th), 'tapas:huge:test', 'un-vec')

fprintf('passed.\n')

%% test: BOLD
fprintf('Testing BOLD generation: ')

n = 1;
th = obj.posterior.mu_n(1,:);
eps = obj.bold_gen( th, obj.data(n), obj.inputs(n), obj.options.hemo, ...
    obj.R, obj.L, obj.idx );
assert(max(abs(eps(:)-obj.trace.epsilon{n}(:)))<1e-14, 'tapas:huge:test', ...
    'bold')

fprintf('passed.\n')

%% test: remove
fprintf('Testing data removal: ')

obj = obj.remove(1:2:10);
assert(obj.N == sum(groupSizes)-5, 'tapas:huge:test', 'remove N');
assert(obj.N == length(obj.data), 'tapas:huge:test', 'data = N');
assert(obj.N == length(obj.inputs), 'tapas:huge:test', 'inputs = N');
assert(isempty(obj.posterior), 'tapas:huge:test', 'remove posterior');


obj = obj.remove('all');
assert(obj.N == 0, 'tapas:huge:test', 'remove N');
assert(obj.N == length(obj.data), 'tapas:huge:test', 'data = N');
assert(obj.N == length(obj.inputs), 'tapas:huge:test', 'inputs = N');

fprintf('passed.\n')

%% test: aux
fprintf('Testing auxiliary functions: ')

assert(tapas_huge_logdet(eye(10)) == 0, 'tapas:huge:test', 'aux logdet');
assert(tapas_huge_logit(.5) == 0, 'tapas:huge:test', 'aux logit');

opts = struct('a',0,'b',0,'c',0);
[ opts ] = tapas_huge_parse_inputs( opts, 'a', 1, 'b', 2 );
assert(all([opts.a==1,opts.b==2,opts.c==0]), 'tapas:huge:test', 'aux nvp');

u = tapas_huge_boxcar( .5, [3;2], [2;1], [.5 1/3], [1.25, .25; .5 0] );
assert(all(u{1}==[0 0 1 1 0 0 1 1 0 0 1 1 0 0 0]'), 'tapas:huge:test',...
    'aux boxcar');
assert(all(u{2}==[0 1 0 1 0 0]'), 'tapas:huge:test', 'aux boxcar');

fprintf('passed.\n')




