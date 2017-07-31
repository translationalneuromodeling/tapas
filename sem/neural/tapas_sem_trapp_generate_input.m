function [input] = tapas_sem_trapp_generate_input(t_sim, theta, ptheta)
%% Generates the input for the model u
%
% Input
%   t_sim   -- Length of the simulation in ms
%   ptheta  -- Structure with fields
%       
% Output
%   input       -- Matrix of dimensions 2 X nnodes X (stime/dt)
%       

% aponteeduardo@gmail.com
% copyright (C) 2017
%
% Integration step

dt = ptheta.dt;

% Coefficients

a_en = theta.a_en;
a_ex = theta.a_ex;

l_en = theta.loc_en; % mm
l_ex = theta.loc_ex; % mm
sigma_in = theta.sigma_in; %mm

% Times

t_fix_off = theta.t_fix_off;
t_ex_on = theta.t_ex_on;
t_ex_off = theta.t_ex_off;

t_ex_delay = theta.t_ex_delay;
t_en_delay = theta.t_en_delay;

tau_ex_on = theta.tau_ex_on;
tau_ex_off = theta.tau_ex_off;

dist = theta.dist;

i_en = a_en * exp(-((dist - l_en) .* (dist - l_en)) / ...
    (2 * sigma_in * sigma_in));
i_ex = a_ex * exp(-((dist - l_ex) .* (dist - l_ex)) / ...
    (2 * sigma_in * sigma_in));

% Generate endogenous

nd = ceil(t_sim/dt);
input = zeros(2, theta.n_nodes, nd);

% Simulate the signal up to the t onset with zeros

%Between on and off
t0 = floor((t_ex_on + t_ex_delay)/dt);
t1 = floor((t_ex_off + t_ex_delay)/dt);

input(1, :,  t0:t1-1) = i_ex * exp(-([t0:(t1 - 1)] - t0) / ( tau_ex_on * dt));
input(1, :, t1:end) = (input(1, t1 - 1) - i_ex) * ...
    exp(-([t1:nd] - t1) / (tau_ex_off * dt));

t0 = floor((t_ex_on + t_en_delay)/dt);
t1 = min(floor((t_ex_off + t_en_delay)/dt), nd);

input(2, :, t0:t1-1) = i_en * ones(1, t1-t0);

input = squeeze(sum(input, 1));
input(end + 1, :) = 1; 

end
