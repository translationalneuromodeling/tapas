function tapas_sem_empirical_delta_plot(congruent, incongruent, quants)
%% Draws the delta plot from a vector of congruent and incongruent responses.
%
% Input
%
%       congruent       -- Vector of congruent reaction times
%       incongruent     -- Vector of incongruent responses
%       quatiles        -- Quantiles used for the delta plot. Defaults to
%                       -- [0.2, 0.4, 0.6, 0.8, 1.0]
%
% Output
%
%       

% aponteeduardo@gmail.com
% copyright (C) 2019
%

n = 2;

n = n + 1;
if nargin < n
    quants = [0.2, 0.4, 0.6, 0.8, 0.97];
end

nq = numel(quants);

qcong = [0 quantile(congruent, quants)];
qincong = [0 quantile(incongruent, quants)];

ycong = zeros(nq, 1);
yincong = zeros(nq, 1);

for i = 1:nq
    ycong(i) = ...
        mean(congruent((qcong(i) < congruent) & (congruent <= qcong(i+1))));
    yincong(i) = ...
        mean(incongruent((qincong(i) < incongruent) & ...
            (incongruent <= qincong(i+1))));
end

plot((ycong + yincong)/2, yincong - ycong, 'k-o');

end
