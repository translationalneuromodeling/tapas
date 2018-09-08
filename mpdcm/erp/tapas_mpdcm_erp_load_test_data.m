function [d] = tapas_mpdcm_erp_load_test_data()
%% 
%
% Input
%
% Output
%

% aponteeduardo@gmail.com
% copyright (C) 2016
%

f = mfilename('fullpath');

[tdir, ~, ~] = fileparts(f);

d = cell(1, 1);

d{1} = load(fullfile(tdir, '..', 'test', 'data', 'derp01.mat'));

d{1} = d{1}.DCM;


end % tapas_mpdcm_erp_load_test_data 

