function [data] = tapas_hgf_data(hgf, pars)
%% Get the date from the hgf. 
%
% Input
%       hgf         -- Complete hgf model.
%       pars        -- Parameters structure
% Output
%       data        -- Structure with the data

% aponteeduardo@gmail.com
% copyright (C) 2016
%

data = struct('y', [], 'u', []);

data.y = hgf.y;
data.u = hgf.u;

end

