function [fits] = tapas_sem_generate_fits(data, samples, model, time)
%% Generate the fits of the data.
%
% Input
%
% Output
%

% aponteeduardo@gmail.com
% copyright (C) 2019
%

CONG = 0;
INCONG = 1;

n = 3;

n = n + 1;
if nargin < n
    t = data.y.t;
    maxt = max(t);
    mint = min(t);

    nv = 300;

    time = linspace(mint * 0.9, maxt * 1.1, nv)';

end

nv = numel(time);

conds = unique(data.u.tt);
nconds = numel(conds);

fits = struct('pro', cell(nconds, 1), 'anti', [], 't', []);

for i = 1:nconds
    fits(i).t = time;

    % Make sure the dimension are correct
    samples = reshape(samples, 1, numel(samples));

    % Iterate over arrays because the interface is cluncky for single times
    for a = [CONG, INCONG]
        llh = zeros(nv, 1);
        for j = 1:nv
            tdata = struct(...
                'y', struct('t', time(j), 'a', a), ...
                'u', struct('tt', conds(i)));
            tllh = model.llh(tdata, samples);
            % Average the log likelihood
            tllh = mean(exp(tllh));
            llh(j) = tllh;
        end
        switch a
            case CONG
                fits(i).pro = llh;
            case INCONG
                fits(i).anti = llh;
        end
    end
end

end
