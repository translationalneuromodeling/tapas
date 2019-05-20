function [fits] = tapas_sem_generate_fits(data, samples, model)
%% Generate the fits of the data.
%
% Input
%
% Output
%

% aponteeduardo@gmail.com
% copyright (C) 2019
%

maxsamples = 30;
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
    ns = numel(samples);

    % Downsample the number of elements!
    spacing = max(1, floor(ns/maxsamples));
    tsamples = samples(1:spacing:end);
    % Use the same input array but different parameters
    tsamples = reshape(tsamples, 1, numel(tsamples));

    % Iterate over arrays because the interface is cluncky for single times
    for a = [0, 1]        
        llh = zeros(nv, 1);
        for j = 1:nv
            tdata = struct(...
                'y', struct('t', pt(j), 'a', a), ...
                'u', struct('tt', conds(i)));
            tllh = model.llh(tdata, tsamples);
            % Average the log likelihood
            tllh = log(mean(exp(tllh)));
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
