pathCogent='C:\fMRI_paradigmas\JHeinzle\Cogent2000v1.32\Toolbox';
pathPsychtoolbox='C:\fMRI_paradigmas\Psychtoolbox\Psychtoolbox';
addpath(pathCogent);
addpath(genpath(pathPsychtoolbox));
rmpath(fullfile(pathPsychtoolbox,'PsychBasic','MatlabWindowsFilesR2007a')); % To get order of path right for PTB
addpath(fullfile(pathPsychtoolbox,'PsychBasic','MatlabWindowsFilesR2007a'));