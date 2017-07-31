function [data] = tapas_sem_multiv_data(data, model, pars)
%% Get the date from the hgf. 
%
% Input
%       hgf         -- Array of complete hgf models.
%       pars        -- Parameters structure
% Output
%       data        -- Structure with the data

% aponteeduardo@gmail.com
% copyright (C) 2016
%

data = data;
ns = size(data, 1);

% Sort the trials to optimize the code
for i = 1:ns
    [~, js] = sort(data(i).y.t);
    data(i).y.t = data(i).y.t(js);
    data(i).y.a = data(i).y.a(js);
    data(i).u.tt = data(i).u.tt(js);
end

end

