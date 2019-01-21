function [fits] = tapas_sem_generate_fits(data, samples, model)
%% 
%
% Input
%
% Output
%

% aponteeduardo@gmail.com
% copyright (C) 2019
%

nv = 100;

t = data.y.t;
conds = unique(data.u.tt);
nconds = numel(conds);

maxt = max(t);
mint = min(t);

pt = linspace(mint * 0.9, maxt * 1.1, nv)';

fits = struct('pro', cell(nconds, 1), 'anti', [], 't', []);

for i = 1:nconds
    fits(i).t = pt;
    tsamples = {mean([samples{:}], 2)};
    for a = [0, 1]        
        llh = zeros(nv, 1);
        for j = 1:nv
            tdata = struct(...
                'y', struct('t', pt(j), 'a', a), ...
                'u', struct('tt', conds(i)));
%            tdata = struct(...
%                'y', struct('t', pt(1:j), 'a', a * ones(j, 1)), ...
%                'u', struct('tt', conds(i) * ones(j, 1)));

            tllh = model.llh(tdata, tsamples);
%            llh(j) = tllh - sum(llh);
            llh(j) = tllh;
        end
        switch a
        case 0
            fits(i).pro = llh;
        case 1
            fits(i).anti = llh;
        end
    end
end

end
