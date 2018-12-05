% HGF v5.2, using the data: PRSSI, EEG, short version of SRL (srl2)
%
% missed trials are removed and therefore not part of the perceptual model
% =========================================================================
% hgfv5_2_srl2
% =========================================================================

function hgfv5_2_srl2_plotOptim

% addpath(genpath('/cluster/project/tnu/igsandra/tapas/'));

%% We load the behavioural srl2 data
data_srl2 = tapas_h2gf_load_example_data_srl2();
% Number of subjects
num_subjects = length(data_srl2);

srl2_estka2 = load('D:\PRSSI\h2gf\results\hgf\estka2\hgf_3l_est_srl2_estka2.mat');
srl2_estka2mu2 = load('D:\PRSSI\h2gf\results\hgf\estka2mu2\hgf_3l_est_srl2_estka2mu2.mat');
srl2_estka2mu3 = load('D:\PRSSI\h2gf\results\hgf\estka2mu3\hgf_3l_est_srl2_estka2mu3.mat');
srl2_estka2sa2 = load('D:\PRSSI\h2gf\results\hgf\estka2sa2\hgf_3l_est_srl2_estka2sa2.mat');
srl2_estka2sa3 = load('D:\PRSSI\h2gf\results\hgf\estka2sa3\hgf_3l_est_srl2_estka2sa3.mat');
srl2_estka2om3 = load('D:\PRSSI\h2gf\results\hgf\estka2om3\hgf_3l_est_srl2_estka2om3.mat');

srl2_estom2 = load('D:\PRSSI\h2gf\results\hgf\estom2\hgf_3l_est_srl2_estom2.mat');
srl2_estom2mu2 = load('D:\PRSSI\h2gf\results\hgf\estom2mu2\hgf_3l_est_srl2_estom2mu2.mat');
srl2_estom2mu3 = load('D:\PRSSI\h2gf\results\hgf\estom2mu3\hgf_3l_est_srl2_estom2mu3.mat');
srl2_estom2sa2 = load('D:\PRSSI\h2gf\results\hgf\estom2sa2\hgf_3l_est_srl2_estom2sa2.mat');
srl2_estom2sa3 = load('D:\PRSSI\h2gf\results\hgf\estom2sa3\hgf_3l_est_srl2_estom2sa3.mat');
srl2_estom2om3 = load('D:\PRSSI\h2gf\results\hgf\estom2om3\hgf_3l_est_srl2_estom2om3.mat');

srl2_est2l = load('D:\PRSSI\h2gf_hgf\hgf_

%%loop through subjects
for i = 1:num_subjects
accu_estka2(i,1)=srl2_estka2.hgf_est_srl2(1,i).optim.accu;
accu_estka2mu2(i,1)=srl2_estka2mu2.hgf_est_srl2(1,i).optim.accu;
accu_estka2mu3(i,1)=srl2_estka2mu3.hgf_est_srl2(1,i).optim.accu;
accu_estka2sa2(i,1)=srl2_estka2sa2.hgf_est_srl2(1,i).optim.accu;
accu_estka2sa3(i,1)=srl2_estka2sa3.hgf_est_srl2(1,i).optim.accu;
accu_estka2om3(i,1)=srl2_estka2om3.hgf_est_srl2(1,i).optim.accu;
accu_estom2(i,1)=srl2_estom2.hgf_est_srl2(1,i).optim.accu;
accu_estom2mu2(i,1)=srl2_estom2mu2.hgf_est_srl2(1,i).optim.accu;
accu_estom2mu3(i,1)=srl2_estom2mu3.hgf_est_srl2(1,i).optim.accu;
accu_estom2sa2(i,1)=srl2_estom2sa2.hgf_est_srl2(1,i).optim.accu;
accu_estom2sa3(i,1)=srl2_estom2sa3.hgf_est_srl2(1,i).optim.accu;
accu_estom2om3(i,1)=srl2_estom2om3.hgf_est_srl2(1,i).optim.accu;

comp_estka2(i,1)=srl2_estka2.hgf_est_srl2(1,i).optim.comp;
comp_estka2mu2(i,1)=srl2_estka2mu2.hgf_est_srl2(1,i).optim.comp;
comp_estka2mu3(i,1)=srl2_estka2mu3.hgf_est_srl2(1,i).optim.comp;
comp_estka2sa2(i,1)=srl2_estka2sa2.hgf_est_srl2(1,i).optim.comp;
comp_estka2sa3(i,1)=srl2_estka2sa3.hgf_est_srl2(1,i).optim.comp;
comp_estka2om3(i,1)=srl2_estka2om3.hgf_est_srl2(1,i).optim.comp;
comp_estom2(i,1)=srl2_estom2.hgf_est_srl2(1,i).optim.comp;
comp_estom2mu2(i,1)=srl2_estom2mu2.hgf_est_srl2(1,i).optim.comp;
comp_estom2mu3(i,1)=srl2_estom2mu3.hgf_est_srl2(1,i).optim.comp;
comp_estom2sa2(i,1)=srl2_estom2sa2.hgf_est_srl2(1,i).optim.comp;
comp_estom2sa3(i,1)=srl2_estom2sa3.hgf_est_srl2(1,i).optim.comp;
comp_estom2om3(i,1)=srl2_estom2om3.hgf_est_srl2(1,i).optim.comp;

LME_estka2(i,1)=srl2_estka2.hgf_est_srl2(1,i).optim.LME;
LME_estka2mu2(i,1)=srl2_estka2mu2.hgf_est_srl2(1,i).optim.LME;
LME_estka2mu3(i,1)=srl2_estka2mu3.hgf_est_srl2(1,i).optim.LME;
LME_estka2sa2(i,1)=srl2_estka2sa2.hgf_est_srl2(1,i).optim.LME;
LME_estka2sa3(i,1)=srl2_estka2sa3.hgf_est_srl2(1,i).optim.LME;
LME_estka2om3(i,1)=srl2_estka2om3.hgf_est_srl2(1,i).optim.LME;
LME_estom2(i,1)=srl2_estom2.hgf_est_srl2(1,i).optim.LME;
LME_estom2mu2(i,1)=srl2_estom2mu2.hgf_est_srl2(1,i).optim.LME;
LME_estom2mu3(i,1)=srl2_estom2mu3.hgf_est_srl2(1,i).optim.LME;
LME_estom2sa2(i,1)=srl2_estom2sa2.hgf_est_srl2(1,i).optim.LME;
LME_estom2sa3(i,1)=srl2_estom2sa3.hgf_est_srl2(1,i).optim.LME;
LME_estom2om3(i,1)=srl2_estom2om3.hgf_est_srl2(1,i).optim.LME;

end

accu_all = [comp_estka2 comp_estka2mu2 comp_estka2mu3 comp_estka2sa2 comp_estka2sa3 comp_estka2om3 ...
    comp_estom2 comp_estom2mu2 comp_estom2mu3 comp_estom2sa2 comp_estom2sa3 comp_estom2om3];
accu_mean = [mean(accu_estka2) mean(accu_estka2mu2) mean(accu_estka2mu3) mean(accu_estka2sa2) mean(accu_estka2sa3) mean(accu_estka2om3) ...
    mean(accu_estom2) mean(accu_estom2mu2) mean(accu_estom2mu3) mean(accu_estom2sa2) mean(accu_estom2sa3) mean(accu_estom2om3)];

comp_all = [comp_estka2 comp_estka2mu2 comp_estka2mu3 comp_estka2sa2 comp_estka2sa3 comp_estka2om3 ...
    comp_estom2 comp_estom2mu2 comp_estom2mu3 comp_estom2sa2 comp_estom2sa3 comp_estom2om3];
comp_mean = [mean(comp_estka2) mean(comp_estka2mu2) mean(comp_estka2mu3) mean(comp_estka2sa2) mean(comp_estka2sa3) mean(comp_estka2om3) ...
    mean(comp_estom2) mean(comp_estom2mu2) mean(comp_estom2mu3) mean(comp_estom2sa2) mean(comp_estom2sa3) mean(comp_estom2om3)];

LME_all = [LME_estka2 LME_estka2mu2 LME_estka2mu3 LME_estka2sa2 LME_estka2sa3 LME_estka2om3 ...
    LME_estom2 LME_estom2mu2 LME_estom2mu3 LME_estom2sa2 LME_estom2sa3 LME_estom2om3];
LME_mean = [mean(LME_estka2) mean(LME_estka2mu2) mean(LME_estka2mu3) mean(LME_estka2sa2) mean(LME_estka2sa3) mean(LME_estka2om3) ...
    mean(LME_estom2) mean(LME_estom2mu2) mean(LME_estom2mu3) mean(LME_estom2sa2) mean(LME_estom2sa3) mean(LME_estom2om3)];


fig1=figure('rend','painters','pos',[10 10 1300 700],'Name',['Optim'],'NumberTitle','off');
subplot(1,3,1); bar(accu_mean);hold on;
title('accuracy')
label_x = {'ka2' 'ka2mu2' 'ka2mu3' 'ka2sa2' 'ka2sa3' 'ka2om3' 'om2' 'om2mu2' 'om2mu3' 'om2sa2' 'om2sa3' 'om2om3'};
ax = gca;
ax.XTickLabel = label_x;
ax.XTick = [1:1:12]; hold on;
xtickangle(45);

subplot(1,3,2); bar(comp_mean);hold on;
title('complexity')
label_x = {'ka2' 'ka2mu2' 'ka2mu3' 'ka2sa2' 'ka2sa3' 'ka2om3' 'om2' 'om2mu2' 'om2mu3' 'om2sa2' 'om2sa3' 'om2om3'};
ax = gca;
ax.XTickLabel = label_x;
ax.XTick = [1:1:12]; hold on;
xtickangle(45);

subplot(1,3,3); bar(LME_mean);hold on;
title('LME')
label_x = {'ka2' 'ka2mu2' 'ka2mu3' 'ka2sa2' 'ka2sa3' 'ka2om3' 'om2' 'om2mu2' 'om2mu3' 'om2sa2' 'om2sa3' 'om2om3'};
ax = gca;
ax.XTickLabel = label_x;
ax.XTick = [1:1:12]; hold on;
xtickangle(45);

[alpha,exp_r,xp,pxp] =spm_BMS(LME_all)
figure; subplot(1,2,1);bar(exp_r);title('exp_r'); subplot(1,2,2); bar(xp); title('xp');
%% We loop through the config files:
for config_file = 1:12
    % Enter the configuration of the binary hgf
    if config_file == 1
        config = 'tapas_hgf_binary_config_estka2_new';
        configtype = 'estka2';
        
    elseif config_file == 2
        config = 'tapas_hgf_binary_config_estka2mu2_new';
        configtype = 'estka2mu2';
    elseif config_file == 3
        config = 'tapas_hgf_binary_config_estka2mu3_new';
        configtype = 'estka2mu3';
    elseif config_file == 4
        config = 'tapas_hgf_binary_config_estka2om3_new';
        configtype = 'estka2om3';
    elseif config_file == 5
        config = 'tapas_hgf_binary_config_estka2sa2_new';
        configtype = 'estka2sa2';
    elseif config_file == 6
        config = 'tapas_hgf_binary_config_estka2sa3_new';
        configtype = 'estka2sa3';
    elseif config_file == 7
        config = 'tapas_hgf_binary_config_estom2_new';
        configtype = 'estom2';
    elseif config_file == 8
        config = 'tapas_hgf_binary_config_estom2mu2_new';
        configtype = 'estom2mu2';
    elseif config_file == 9
        config = 'tapas_hgf_binary_config_estom2mu3_new';
        configtype = 'estom2mu3';
    elseif config_file == 10
        config = 'tapas_hgf_binary_config_estom2om3_new';
        configtype = 'estom2om3';
    elseif config_file == 11
        config = 'tapas_hgf_binary_config_estom2sa2_new';
        configtype = 'estom2sa2';
    elseif config_file == 12
        config = 'tapas_hgf_binary_config_estom2sa3_new';
        configtype = 'estom2sa3';
    elseif config_file == 13
        config = 'tapas_rw_binary_config';
        configtype = 'estrw';
    end
end

%% define where to store results:
f = mfilename('fullpath');

[tdir, ~, ~] = fileparts(f);

maskResFolder = ([tdir,'/results/hgf/',configtype]);
mkdir(maskResFolder);
cd (maskResFolder);
for id = 1:num_subjects
    disp(['Scan #',num2str(id), ': ', configtype]);
    disp('***************************************');
    
    % run inference
    hgf_est_srl2(id) = tapas_fitModel(data_srl2(id).y, data_srl2(id).u, config, 'tapas_unitsq_sgm_config', 'tapas_quasinewton_optim_config');
    
    % save data and plot trajectories:
    if config_file == 13
        save(['hgf_rw_est_srl2_',configtype,'.mat'],'hgf_est_srl2'); 
        tapas_rw_binary_plotTraj(hgf_est_srl2(id));
        print(['srl2_re_hgf_rw_',configtype,'_subjnr_',num2str(id)],'-dtiff');   
    else
        save(['hgf_3l_est_srl2_',configtype,'.mat'],'hgf_est_srl2');
        tapas_hgf_binary_plotTraj(hgf_est_srl2(id));
        print(['srl2_re_hgf_3l_',configtype,'_subjnr_',num2str(id)],'-dtiff'); 
    end
    delete(findall(0,'Type','figure'));
end
end