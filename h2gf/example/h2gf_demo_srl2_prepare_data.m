% SRL behavioral data analyses
% This script creates the input vectors for the hgf from the raw behavioral data.
% 4 different models are computed (identical to Neuron paper 2013):
% - HGF_1_fixom: three level HGF with fix omega
% - HGF_3_fixth_fixka: two level HGF
% - RW_1: Rescorla-Wagner Model
% - Sutton_1: reinforcement learning model with variable learning rate
%
% missed trials are modeled as 'NaN', and are part of the perceptual model
% =========================================================================
% 15-04-2016; Sandra Iglesias
% 27-11-2017; Sandra Iglesias
% for i in {1..80}; do bsub -W 0:54 -o o_hgf_$i matlab -singleCompThread -r "bp_hgf_srl1_hgfv5_1($i,m)";done
% =========================================================================

function h2gf_demo_srl2_prepare_data
addpath('/cluster/project/tnu/igsandra/HGF_sim/h2gf/code_srl_SSP/code')
addpath(genpath('/cluster/project/tnu/igsandra/HGF_sim/h2gf/code_srl_SSP/'))
addpath(genpath('/cluster/project/tnu/igsandra/HGF_sim/h2gf/code_srl_low40_500'))
addpath(genpath('/cluster/project/tnu/igsandra/HGF_sim/h2gf/code_srl_SSP/code'))
addpath (genpath('/cluster/project/tnu/igsandra/HGF_sim/h2gf/tapas/external/'))
addpath ('/cluster/project/tnu/igsandra/HGF_sim/h2gf/');
addpath '/cluster/project/tnu/igsandra/HGF_sim/h2gf/tapas/tools/ti/linear/'
addpath '/cluster/project/tnu/igsandra/HGF_sim/h2gf/tapas/tools/ti/'
options = prssi_set_analysis_options_srl2;
clear SRL

maskModel = {'HGF_1_fixom_v5_1'};

disp(['This is hgfToolBox_v5.1 srl EEG study 2:', maskModel]);% Go through scans


%% prepare data
% Number of subjects
num_subjects = length(options.subjectIDs);
disp(['number of subjects: ', num_subjects])

%randomly assign data position
subjPosition = randperm(num_subjects);

% Initialize a structure for the data
data_srl2 = struct('y', cell(num_subjects, 1), ...
    'u', cell(num_subjects, 1));

subjindex = 0;
for idCell = options.subjectIDs
    subjindex = subjindex+1;
    subjindexPosition = subjPosition(subjindex);
    
    disp('_________________________________________')
    id = char(idCell)
    disp('_________________________________________');
    details = prssi_subjects_srl2(id);
    behav = load([details.behavroot '/' details.subjname '.mat']);
    
    irr = [];
    for l = 1:length(behav.SRL.Concatenate(:,3))
        if behav.SRL.Concatenate(l,4) == -99 %|| behav.SSL.Concatenate(l,3) > 1000
            irr = [irr, l];
        end
    end
    SRL.Re.irr = irr;
    if isempty(irr)
        irrout = 'none';
    else
        irrout = irr;
    end
    disp(['missed trials:', num2str(irrout)]);
    
    irr_late = [];
    for l = 1:length(behav.SRL.Concatenate(:,3))
        if behav.SRL.Concatenate(l,3) > 1500
            irr_late = [irr_late, l];
        end
    end
    SRL.Re.irr_late = irr_late;
    if isempty(irr_late)
        irrout_late = 'none';
    else
        irrout_late = irr_late;
    end
    disp(['late trials:', num2str(irrout_late)]);
    
    
    SRL.Re.corrects = behav.SRL.Concatenate(:,2);
    for trials = 1:length(behav.SRL.Concatenate(:,3))
        if behav.SRL.Concatenate(trials,1) == 0 % layout: yellow left
            if behav.SRL.Concatenate(trials,4) == 2
                SRL.Re.observed_choices(trials,1) = 0;
            else
                SRL.Re.observed_choices(trials,1) = 1;
            end
        elseif behav.SRL.Concatenate(trials,1) == 1 % layout: yellow right
            if behav.SRL.Concatenate(trials,4) == 2
                SRL.Re.observed_choices(trials,1) = 1;
            else
                SRL.Re.observed_choices(trials,1) = 0;
            end
        end
    end
    
    SRL.Re.corrects(SRL.Re.irr)=[];
    SRL.Re.observed_choices(SRL.Re.irr)=[];
    
    % Fill the responses
    data_srl2(subjindexPosition).y = SRL.Re.observed_choices(:,1);
    % and experimental manipulations
    data_srl2(subjindexPosition).u = SRL.Re.corrects(:,1);
    cd([options.maincodedir]);
    save('data_srl2.mat','data_srl2');
end
load('data_srl2.mat');
data_srl2(subjPosition(13)) = [];
save('data_srl2.mat','data_srl2');
end