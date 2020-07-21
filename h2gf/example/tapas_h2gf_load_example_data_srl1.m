function [data_srl1] = tapas_h2gf_load_example_data_srl1()
%% Return an examplary response and input  
%
% Input
%   
% Output
%   y       An exemplary response of a binary hgf
%   u       An exemplary input to the hgf
%

% aponteeduardo@gmail.com
% copyright (C) 2017
%

% Get current location
f = mfilename('fullpath');

[tdir, ~, ~] = fileparts(f);

data = load(fullfile(tdir, 'data_srl1.mat'));
data_srl1 = data.data_srl1;

end

