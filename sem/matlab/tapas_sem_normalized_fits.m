function [fits] = tapas_sem_normalized_fits(data, fits)
%% Normalize the fits according to the data.
%
% Input
%       data        -- Data structure with field y and u
%       fits        -- Fits structure with fields pro and anti
%
% Output
%       fits        -- Normalized fit structure

% aponteeduardo@gmail.com
% copyright (C) 2019
%

% Number of conditions
conds = unique(data.u.tt);

for i = 1:numel(conds)
    % Number of trials per condition
    nt = sum(data.u.tt == conds(i));
    fits(i).pro = fits(i).pro * nt;
    fits(i).anti = fits(i).anti * nt;
end

end
