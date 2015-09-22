function tapas_fit_plotCorr(r)
% Plots the posterior correlation matrix of the model parameters estimated by tapas_fitModel.
% This is estimated by calculating the Hessian at the MAP estimate. The negative inverse of the
% Hessian is the parameter covariance, which is standardized to yield the correlation.
% Usage:  est = tapas_fitModel(responses, inputs); tapas_fit_plotCorr(est);
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Set up display
scrsz = get(0,'screenSize');
outerpos = [0*scrsz(3),0.4*scrsz(4),0.6*scrsz(4),0.6*scrsz(4)];
figure(...
    'OuterPosition', outerpos,...
    'Name','Posterior parameter correlation matrix');

% Determine indices of parameters to optimize (i.e., those that are not fixed and not NaN)
prc_ind = r.c_prc.priorsas;
prc_ind(isnan(prc_ind)) = 0;
prc_ind = find(prc_ind);

obs_ind = r.c_obs.priorsas;
obs_ind(isnan(obs_ind)) = 0;
obs_ind = find(obs_ind);

n_par   = length(prc_ind) + length(obs_ind);

% Find names of optimized parameters to use them as tick labels 
names_prc = fieldnames(r.p_prc);
fields = struct2cell(r.p_prc);
expnms_prc = [];
for k = 1:length(names_prc)
    for l= 1:length(fields{k})
    expnms_prc = [expnms_prc, names_prc(k)];
    end
end
expnms_prc = expnms_prc(1:length(r.p_prc.p))';

names_obs = fieldnames(r.p_obs);
fields = struct2cell(r.p_obs);
expnms_obs = [];
for k = 1:length(names_obs)
    for l= 1:length(fields{k})
    expnms_obs = [expnms_obs, names_obs(k)];
    end
end
expnms_obs = expnms_obs(1:length(r.p_obs.p))';

ticklabels = {[expnms_prc(prc_ind); expnms_obs(obs_ind)]};

% Plot matrix
imagesc(r.optim.Corr, [-1 1]);
axis('square');
set(gca,'xtick',1:n_par);
set(gca,'ytick',1:n_par);
set(gca,'xticklabel',ticklabels{:});
set(gca,'yticklabel',ticklabels{:});
colorbar;
title('Parameter correlation', 'FontSize', 15, 'FontWeight', 'bold');
