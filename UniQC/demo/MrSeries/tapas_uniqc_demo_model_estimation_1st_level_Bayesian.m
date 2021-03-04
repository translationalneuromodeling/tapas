% Script demo_bayes
% 1st level model specification and estimation using a Bayesian estimation
%
%  demo_bayes
%
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-05-04
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


clear;
close all;
clc;
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (1) Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% uses the output of demo_model_estimation_1st_level
S = MrSeries('/data/home/uqsboll2/code/uniqc-code/demo/MrSeries/model_estimation/MrSeries_210226_170649');
% change directory to get a separate it from the preprocessing
S.parameters.save.path = strrep(S.parameters.save.path, 'model_estimation', 'model_estimation_bayes');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (2) Specify Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% swith to bayes
S.glm.estimationMethod = 'Bayesian';
% contrasts need to be specified already here
S.glm.gcon(1).name = 'tapping';
S.glm.gcon(1).convec = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (3) Estimate Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute stat images
S.compute_stat_images;
% estimate
S.specify_and_estimate_1st_level;