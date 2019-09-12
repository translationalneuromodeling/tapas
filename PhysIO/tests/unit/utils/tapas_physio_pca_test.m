function tests = tapas_physio_pca_test()
% Test both pca() from "stats" toolbox and svd() built-in Matlab functions
% give the same results.
%
%   tests = tapas_physio_pca_test()
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_pca_test
%
% See also tapas_physio_pca

tests = functiontests(localfunctions);

end % function

function test_pca_more_voxels_than_volumes(testCase)
% Compare the outputs of tapas_physio_pca(X,'svd') and
% tapas_physio_pca(X,'stats-pca') and check is they are close enough
% numericaly, within a certain tolerence.

% Reset the Random Number Generator
rng('default')

% Generate timeseries
nVoxels  = 10000;
NVolumes = 300;
timeseries = randn(nVoxels,NVolumes);

% Perform PCA with both methods
svd   = struct;
stats = struct;
[svd.  COEFF, svd.  SCORE, svd.  LATENT, svd.  EXPLAINED, svd.  MU] = tapas_physio_pca( timeseries, 'svd'       );
[stats.COEFF, stats.SCORE, stats.LATENT, stats.EXPLAINED, stats.MU] = tapas_physio_pca( timeseries, 'stats-pca' );

% Compare both methods
verifyEqual(testCase,svd.COEFF    ,stats.COEFF    )
verifyEqual(testCase,svd.SCORE    ,stats.SCORE    )
verifyEqual(testCase,svd.LATENT   ,stats.LATENT   )
verifyEqual(testCase,svd.EXPLAINED,stats.EXPLAINED)
verifyEqual(testCase,svd.MU       ,stats.MU       )

end % function

function test_pca_more_volumes_than_voxels(testCase)
% Compare the outputs of tapas_physio_pca(X,'svd') and
% tapas_physio_pca(X,'stats-pca') and check is they are close enough
% numericaly, within a certain tolerence.

% Reset the Random Number Generator
rng('default')

% Generate timeseries
nVoxels  = 200;
NVolumes = 300;
timeseries = randn(nVoxels,NVolumes);

% Perform PCA with both methods
verifyError(testCase,@() tapas_physio_pca( timeseries, 'svd'       ) ,'tapas_physio_pca:NotEnoughVoxels')
verifyError(testCase,@() tapas_physio_pca( timeseries, 'stats-pca' ) ,'tapas_physio_pca:NotEnoughVoxels')

end % function
