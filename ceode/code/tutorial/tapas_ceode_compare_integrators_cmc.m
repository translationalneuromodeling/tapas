function [ ] = tapas_ceode_compare_integrators_cmc()
% [  ] = tapas_ceode_compare_integrators_cmc()
%
% Takes a synthetic dataset (3 population ERP model), and computes the 
% predicted response for different integrators and levels of delays. The 
% continuous extension for ODE method (ceode)is taken as reference.
%
% INPUT
%
% OUTPUT
%
%
% -------------------------------------------------------------------------
%
% Author: Dario Schöbi
% Created: 2020-08-10
% Copyright (C) 2020 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS ceode Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
% -------------------------------------------------------------------------


% Load DCM template
DCM = tapas_ceode_synData_cmc();

% Integrators
intSpec  = {...
        'spm_int_L', 'spm_fx_cmc'; ...
    'tapas_ceode_int_euler', 'tapas_ceode_fx_cmc'};

% Sets of Delays
tau = linspace(-1, -0.5, 10);

% Pyramidal cell voltage states
vPyramids = [7, 8];

% Preallocate signals
x = cell(length(intSpec), 1);
for i = 1 : length(x)
    x{i} = zeros(length(tau), length(vPyramids) * DCM.M.ns);
end


%------------------------- GENERATE PREDICTIONS ---------------------------

% Integrate the DDE with the different integrators
tau_idx = 1;
for delays = tau
    
    % Add delay to forward connection (read as FROM (columns) TO (rows)
    DCM.Ep.D(2, 1) = delays;
    
    % Iterate over integrator specs in DCM structure and generate signal
    for i = 1 : size(intSpec, 1)
        DCM.M.int = intSpec{i, 1};
        DCM.M.f = intSpec{i, 2};
        
        y = tapas_ceode_gen_erp(DCM.Ep, DCM.M, DCM.xU);
        x{i}(tau_idx, :) = spm_vec(y{1}(:, vPyramids));
    end

    % Increase index for storing signal in single structure
    tau_idx = tau_idx + 1;
end


%---------------------------- VISUALIZATION--------------------------------
figure(); 

% Color code
cc = jet(length(tau));

% Plot the predicted pyramidal voltages
nIntegrators = length(x);

for i = 1 : nIntegrators

    subplot(2, nIntegrators, i);
    hold on;
    title(intSpec{i, 1}, 'Interpreter', 'none');
    
    % Plot the predictions
    for j = 1 : length(tau)
        y = reshape(x{i}(j, :), [], length(vPyramids));
        plot_prediction(DCM.xU.dt .* [0 : DCM.M.ns-1], y, cc(j, :));

    end
    
    % Configure axes to create a nice plot
    config_prediction(gca, 'raw');
        
end

% Plot the difference to last integration scheme
for i = 1 : nIntegrators-1
    
    subplot(2, nIntegrators, i + nIntegrators);
    hold on;
    title([intSpec{i, 1} ' - ' intSpec{end, 1}], 'Interpreter', 'none');

    % Plot signal difference
    for j = 1 : length(tau)
        y = reshape(x{i}(j, :) - x{nIntegrators}(j, :), ...
            [], length(vPyramids));
        plot_prediction(DCM.xU.dt .* [0 : DCM.M.ns-1], y, cc(j, :));
    end
    
    % Configure axes to create a nice plot
    config_prediction(gca, 'delta');
    
end

% Plot regression coefficient

% Time samples to compute pearson correlation (only the signal from the
% second (delayed region is used))
rsamples = DCM.M.ns + 1 : 2 * DCM.M.ns;

subplot(2, nIntegrators, 2 * nIntegrators);
hold on;

xx = 1;
tickLabels = {};
for i = 1 : nIntegrators-1
    
    % X-Axis offset for dot-plot
    xoffs = xx + linspace(-0.2, 0.2, length(tau));
    
    for j = 1 : length(tau)
        
        rCoeff = corrcoef(x{i}(j, rsamples), x{nIntegrators}(j, rsamples));
        plot(xoffs(j), rCoeff(1, 2), 'o', ...
            'MarkerFaceColor', cc(j, :), ...
            'MarkerEdgeColor', 'none', ...
            'MarkerSize', 10);
    end
    
    xx = xx + 1;
    
end

% Configure the axes of the regression plot
config_regression(gca, xoffs, round(8 * exp(tau), 1));

end


% ------------------------- CONFIGURATIONS--------------------------------

function plot_prediction(t, y, cc)
% Plot the predictions from the two regions

h = plot(t, y,...
    'Color', cc);
h(1).LineStyle = '--';
h(2).LineStyle = '-';

end

function config_prediction(h, plotFlag)
% Configure the prediction plot (prediction and difference)

set(h, 'TickDir', 'out', ...
    'XGrid', 'on', ...
    'XLim', [0, 0.51], ...
    'XTick', [0 : 0.1 : 0.5], ...
    'YGrid', 'on', ...
    'PlotBoxAspectRatio', [1 1 1], ...
    'FontSize', 14);

% Set axes labels
h.XLabel.String = 'Time [s]';
switch plotFlag
    case 'raw'
        h.YLabel.String = 'predicted signal (y_p)';
    case 'delta'
        h.YLabel.String = '\Delta y_p';
end

% Increase LineWidth
hh = findobj(h.Children, 'Type', 'Line');
set(hh, 'LineWidth', 2);

end


function config_regression(h, xticks, tickLabels)
% Configure the regression plot

set(h, 'XTick', xticks, ...
    'XTickLabels', tickLabels, ...
    'XTickLabelRotation', 45, ...
    'XGrid', 'on', ...
    'YGrid', 'on', ...
    'PlotBoxAspectRatio', [1 1 1]);


h.YLabel.String = 'Pearson Correlation btw. integrators';
h.XLabel.String = 'Delays [ms]';

end
