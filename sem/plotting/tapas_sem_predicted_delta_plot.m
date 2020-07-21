function [x, y] = tapas_sem_predicted_delta_plot(time, congruent, ...
    incongruent, quants)
%% Plot the predicted delta plots 
%
% Input
%       time        -- Vector of time.
%       congruent   -- Vector of likelihood of congruent responses
%       incongruen  -- Vector of likelihood of incongruent responses
%
% Output
%

% aponteeduardo@gmail.com
% copyright (C) 2019
%

n = 3;

n = n + 1;
if nargin < n
    quants = [0.2, 0.4, 0.6, 0.8, 0.98];
end


nq = numel(quants);

zcong = cumtrapz(time, congruent);
cong = congruent/zcong(end);
zcong = zcong/zcong(end);

zincong = cumtrapz(time, incongruent);
incong = incongruent/zincong(end);
zincong = zincong/zincong(end);

yincong = zeros(nq, 1);
ycong = zeros(nq, 1);


q0 = 0;

for i = 1:nq
    i_cong = (q0 < zcong) & (zcong <= quants(i));
    ycong(i) = trapz(time, time .* cong .* i_cong) ./ (quants(i) - q0);

    i_incong = (q0 < zincong) & (zincong <= quants(i));
    yincong(i) = trapz(time, time .* incong .* i_incong) ./ (quants(i) - q0);
   
    q0 = quants(i); 
end

x = (ycong + yincong)/2;
y = (yincong - ycong);

plot(x, y, 'o--b')

end
