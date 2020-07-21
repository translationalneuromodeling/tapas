function [y, u] = tapas_h2gf_load_example_data()
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

data = load(fullfile(tdir, 'data', 'example_h2gf.mat'));
y = data.y;
u = data.u;

end

