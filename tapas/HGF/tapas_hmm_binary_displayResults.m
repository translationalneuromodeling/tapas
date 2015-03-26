function tapas_hmm_binary_displayResults(r)
% Displays results estimated by fitModel for the hmm perceptual model in the binary case
% Usage:  est = fitModel(responses, inputs); hmm_binary_displayResults(est);
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Print transition matrix
Ared  = r.p_prc.Ared';
Alast = 1 - Ared;
A = [Ared, Alast];

disp(' ')
disp('Transition matrix A: ')
disp(' ')
disp(A)

% Print state prior
ppired  = r.p_prc.ppired;
ppilast = 1 - ppired;
ppi = [ppired, ppilast];

disp(' ')
disp('State prior pi: ')
disp(' ')
disp(ppi)

% Set up display
scrsz = get(0,'screenSize');
outerpos = [0.2*scrsz(3),0.7*scrsz(4),0.8*scrsz(3),0.3*scrsz(4)];
figure(...
    'OuterPosition', outerpos,...
    'Name','RW binary fit results');

% Inputs
u = r.u(:,1)-1;

% Responses
y = r.y;

% Number of trials
t = length(u);

% Plot
plot(0:t, [ppi(1); r.traj.alpr(:,1)], 'r', 'LineWidth', 2);
hold all;
plot(0, ppi(1), 'or', 'LineWidth', 2); % prior
plot(1:t, u, '.', 'Color', [0 0.6 0]); % inputs
if ~isempty(find(strcmp(fieldnames(r),'y'))) && ~isempty(r.y)
    if ~isempty(find(strcmp(fieldnames(r),'irr')))
        y(r.irr) = NaN; % weed out irregular responses
    end
    plot(1:t, y, '.', 'Color', [1 0.7 0]); % responses
    title(['Response y (orange), input u (green), and posterior probability \alpha'' (red) of option 1'], 'FontWeight', 'bold');
    ylabel('y, u, \alpha''');
    axis([0 t -0.15 1.15]);
else
    title(['Input u (green) and posterior probability \alpha'' (red) of option 1'], 'FontWeight', 'bold');
    ylabel('u, \alpha''');
    axis([0 t -0.1 1.1]);
end
plot(1:t, 0.5, 'k');
xlabel('Trial number');
hold off;
