% h2gf demo using the data: PRSSI, EEG, long version of SRL (srl1)
%
% plot boxplots for all parameters and LME
% =========================================================================
% h2gf_demo_srl1_summary(4000,1,1)
% =========================================================================

function h2gf_demo_srl1_summary(NrIter,spec_eta,config_file)

addpath(genpath('/cluster/project/tnu/igsandra/tapas/'));

maskModel = {'HGF_1_fixom_v5_1'};
%% specify eta_label
eta_label = num2str(spec_eta);

%% specify which configuration of the binary hgf has been used
if config_file == 1
    configtype = 'estka2';
elseif config_file == 2
    configtype = 'estka2mu2';
elseif config_file == 3
    configtype = 'estka2mu3';
elseif config_file == 4
    configtype = 'estka2om3';
elseif config_file == 5
    configtype = 'estka2sa2';
elseif config_file == 6
    configtype = 'estka2sa3';
elseif config_file == 7
    configtype = 'estom2';
elseif config_file == 8
    configtype = 'estom2mu2';
elseif config_file == 9
    configtype = 'estom2mu3';
elseif config_file == 10
    configtype = 'estom2om3';
elseif config_file == 11
    configtype = 'estom2sa2';
elseif config_file == 12
    configtype = 'estom2sa3';
end

disp(['config file:', configtype]);
disp('**************************************');

%% define where results have been stored:
f = mfilename('fullpath');

[tdir, ~, ~] = fileparts(f);

maskResFolder = ([tdir,'/results/',configtype,'/eta', eta_label,'/', num2str(NrIter)]);
% maskResFolder = (['D:\PRSSI\h2gf\/results/',configtype,'/eta', eta_label,'/', num2str(NrIter)]);
cd(maskResFolder);
% listFiles=dir('*h2gf_3l_fixom_est_srl1*');

disp(['This is stats h2gf: ', maskModel{1}]);

for m = 1:12 %length(listFiles)
    clear h2gf_inf;
    % Find data:
    disp(['This is inference Nr.: (',mat2str(m),')']);

    h2gf_inf = load([maskResFolder, '/h2gf_3l_est_srl1_',configtype,'_eta',eta_label,'_', num2str(NrIter),'_',num2str(m),'.mat']);
    
    for s = 1:length(h2gf_inf.summary)
        
        %% estimated parameters
        srl1_estpar_h2gf.param.p_prc = h2gf_inf.summary(s).p_prc;
        srl1_estpar_h2gf.param.p_obs = h2gf_inf.summary(s).p_obs;
        srl1_estpar_h2gf.param.LME = h2gf_inf.summary(s).optim.LME;
        
        srl1_estpar_h2gf.param.mu2_0 = srl1_estpar_h2gf.param.p_prc.mu_0(1,2);
        srl1_estpar_h2gf.param.mu3_0 = srl1_estpar_h2gf.param.p_prc.mu_0(1,3);
        srl1_estpar_h2gf.param.sa2_0 = srl1_estpar_h2gf.param.p_prc.sa_0(1,2);
        srl1_estpar_h2gf.param.sa3_0 = srl1_estpar_h2gf.param.p_prc.sa_0(1,3);
        
        srl1_estpar_h2gf.param.ka = srl1_estpar_h2gf.param.p_prc.ka(1,2);
        srl1_estpar_h2gf.param.om2 = srl1_estpar_h2gf.param.p_prc.om(1,2);
        srl1_estpar_h2gf.param.om3 = srl1_estpar_h2gf.param.p_prc.om(1,3);
        
        srl1_estpar_h2gf.param.ze = srl1_estpar_h2gf.param.p_obs.ze;
        
        % Structure stats for all        
        AllInv_srl1_h2gf.LME(s,m)  = srl1_estpar_h2gf.param.LME;
        
        AllInv_srl1_h2gf.mu2_0(s,m)  = srl1_estpar_h2gf.param.mu2_0;
        AllInv_srl1_h2gf.mu3_0(s,m) = srl1_estpar_h2gf.param.mu3_0;
        AllInv_srl1_h2gf.sa2_0(s,m) = srl1_estpar_h2gf.param.sa2_0;
        AllInv_srl1_h2gf.sa3_0(s,m) = srl1_estpar_h2gf.param.sa3_0;
        
        AllInv_srl1_h2gf.ka(s,m)  = srl1_estpar_h2gf.param.ka;
        AllInv_srl1_h2gf.om2(s,m) = srl1_estpar_h2gf.param.om2;
        AllInv_srl1_h2gf.om3(s,m) = srl1_estpar_h2gf.param.om3;
        AllInv_srl1_h2gf.ze(s,m) = srl1_estpar_h2gf.param.ze;
        
        save (['AllInv_srl1_h2gf_',configtype,'_eta',eta_label,'_', num2str(NrIter),'.mat'], '-struct','AllInv_srl1_h2gf');
        clear srl1_estpar_h2gf;
        
    end
    
end

%%lme
figure('Color',[1 1 1]); hold on;
box_input = AllInv_srl1_h2gf.LME';
figure('Color',[1 1 1]); hold on;
col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
            1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
            0.8 0.0 0.6];
boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
[i,j]=(max(median(box_input(:,1:length(box_input(1,:))))));
mean_LME= mean2(AllInv_srl1_h2gf.LME);
std_LME = std2(AllInv_srl1_h2gf.LME);
plot([0 length(h2gf_inf.summary)+1],[mean_LME mean_LME],'black');
title({['h2gf LME']; ['(mean: ', num2str(mean_LME),'; std: ', num2str(std_LME),')']});
saveas(gcf,['srl1_h2gf_LME_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'fig');
print(['srl1_h2gf_LME_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'-dtiff');


%%kappa
figure('Color',[1 1 1]); hold on;
box_input = AllInv_srl1_h2gf.ka';
prior_index = 11;
figure('Color',[1 1 1]); hold on;
col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
            1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
            0.8 0.0 0.6];
boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
[i,j]=(max(median(box_input(:,1:length(box_input(1,:))))));
mean_ka= mean2(AllInv_srl1_h2gf.ka);
std_ka = std2(AllInv_srl1_h2gf.ka);
if exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1))>min(box_input(:))
    y1 = min(box_input(:))-0.2;
else
    y1 = exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1))-0.2;
end
if exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1))<max(box_input(:))
    y2 = max(box_input(:))+0.2;
else
    y2 = exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1))+0.2;
end
plot([0 length(h2gf_inf.summary)+1],[mean_ka mean_ka],'black'); hold on;
plot([0 length(h2gf_inf.summary)+1],[exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1)) exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1))],'r');
axis([0 length(h2gf_inf.summary)+1 y1 y2])
title({['h2gf ka']; ['(mean: ', num2str(mean_ka),'; std: ', num2str(std_ka),')']});
saveas(gcf,['srl1_h2gf_ka_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'fig');
print(['srl1_h2gf_ka_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'-dtiff');


%%omega2
figure('Color',[1 1 1]); hold on;
box_input = AllInv_srl1_h2gf.om2';
prior_index = 13;
col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
            1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
            0.8 0.0 0.6];
boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
[i,j]=(max(median(box_input(:,1:length(box_input(1,:))))));
mean_om2= mean2(AllInv_srl1_h2gf.om2);
std_om2 = std2(AllInv_srl1_h2gf.om2);

plot([0 length(h2gf_inf.summary)+1],[mean_om2 mean_om2],'black'); hold on;
plot([0 length(h2gf_inf.summary)+1],[h2gf_inf.hgf.c_prc.priormus(prior_index,1) h2gf_inf.hgf.c_prc.priormus(prior_index,1)],'r');

title({['h2gf om2']; ['(mean: ', num2str(mean_om2),'; std: ', num2str(std_om2),')']});
saveas(gcf,['srl1_h2gf_om2_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'fig');
print(['srl1_h2gf_om2_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'-dtiff');

%%omega3
figure('Color',[1 1 1]); hold on;
box_input = AllInv_srl1_h2gf.om3';
prior_index = 14;
if h2gf_inf.hgf.c_prc.priormus(prior_index,1)>min(box_input(:))
    y1 = min(box_input(:))-0.2;
else
    y1 = h2gf_inf.hgf.c_prc.priormus(prior_index,1)-0.2;
end
if h2gf_inf.hgf.c_prc.priormus(prior_index,1)<max(box_input(:))
    y2 = max(box_input(:))+0.2;
else
    y2 = h2gf_inf.hgf.c_prc.priormus(prior_index,1)+0.2;
end
col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
            1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
            0.8 0.0 0.6];
boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
[i,j]=(max(median(box_input(:,1:length(box_input(1,:))))));
mean_om3= mean2(AllInv_srl1_h2gf.om3);
std_om3 = std2(AllInv_srl1_h2gf.om3);
plot([0 length(h2gf_inf.summary)+1],[mean_om3 mean_om3],'black');hold on;
plot([0 length(h2gf_inf.summary)+1],[h2gf_inf.hgf.c_prc.priormus(prior_index,1) h2gf_inf.hgf.c_prc.priormus(prior_index,1)],'r');
axis([0 length(h2gf_inf.summary)+1 y1 y2])
title({['h2gf om3']; ['(mean: ', num2str(mean_om3),'; std: ', num2str(std_om3),')']});
saveas(gcf,['srl1_h2gf_om3_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'fig');
print(['srl1_h2gf_om3_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'-dtiff');

%%mu2_0
figure('Color',[1 1 1]); hold on;
box_input = AllInv_srl1_h2gf.mu2_0';
prior_index = 2;
if h2gf_inf.hgf.c_prc.priormus(prior_index,1)>min(box_input(:))
    y1 = min(box_input(:))-0.2;
else
    y1 = h2gf_inf.hgf.c_prc.priormus(prior_index,1)-0.2;
end
if h2gf_inf.hgf.c_prc.priormus(prior_index,1)<max(box_input(:))
    y2 = max(box_input(:))+0.2;
else
    y2 = h2gf_inf.hgf.c_prc.priormus(prior_index,1)+0.2;
end
col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
            1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
            0.8 0.0 0.6];
boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
[i,j]=(max(median(box_input(:,1:length(box_input(1,:))))));
mean_mu2_0= mean2(AllInv_srl1_h2gf.mu2_0);
std_mu2_0 = std2(AllInv_srl1_h2gf.mu2_0);
plot([0 length(h2gf_inf.summary)+1],[mean_mu2_0 mean_mu2_0],'black');hold on;
plot([0 length(h2gf_inf.summary)+1],[h2gf_inf.hgf.c_prc.priormus(prior_index,1) h2gf_inf.hgf.c_prc.priormus(prior_index,1)],'r');
axis([0 length(h2gf_inf.summary)+1 y1 y2])
title({['h2gf mu2 0']; ['(mean: ', num2str(mean_mu2_0),'; std: ', num2str(std_mu2_0),')']});
saveas(gcf,['srl1_h2gf_mu2_0_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'fig');
print(['srl1_h2gf_mu2_0_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'-dtiff');


%%mu3_0
figure('Color',[1 1 1]); hold on;
box_input = AllInv_srl1_h2gf.mu3_0';
prior_index = 3;
col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
            1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
            0.8 0.0 0.6];
boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
[i,j]=(max(median(box_input(:,1:length(box_input(1,:))))));
mean_mu3_0= mean2(AllInv_srl1_h2gf.mu3_0);
std_mu3_0 = std2(AllInv_srl1_h2gf.mu3_0);
plot([0 length(h2gf_inf.summary)+1],[mean_mu3_0 mean_mu3_0],'black');
plot([0 length(h2gf_inf.summary)+1],[h2gf_inf.hgf.c_prc.priormus(prior_index,1) h2gf_inf.hgf.c_prc.priormus(prior_index,1)],'r');
title({['h2gf mu3 0']; ['(mean: ', num2str(mean_mu3_0),'; std: ', num2str(std_mu3_0),')']});
saveas(gcf,['srl1_h2gf_mu3_0_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'fig');
print(['srl1_h2gf_mu3_0_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'-dtiff');


%%sa2_0
figure('Color',[1 1 1]); hold on;
box_input = AllInv_srl1_h2gf.sa2_0';
prior_index = 5;
if exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1))>min(box_input(:))
    y1 = min(box_input(:))-0.2;
else
    y1 = exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1))-0.2;
end
if exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1))<max(box_input(:))
    y2 = max(box_input(:))+0.2;
else
    y2 = exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1))+0.2;
end
col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
            1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
            0.8 0.0 0.6];
boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
[i,j]=(max(median(box_input(:,1:length(box_input(1,:))))));
mean_sa2_0= mean2(AllInv_srl1_h2gf.sa2_0);
std_sa2_0 = std2(AllInv_srl1_h2gf.sa2_0);
plot([0 length(h2gf_inf.summary)+1],[mean_sa2_0 mean_sa2_0],'black');
plot([0 length(h2gf_inf.summary)+1],[exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1)) exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1))],'r');
axis([0 length(h2gf_inf.summary)+1 y1 y2])
title({['h2gf sa2 0']; ['(mean: ', num2str(mean_sa2_0),'; std: ', num2str(std_sa2_0),')']});
saveas(gcf,['srl1_h2gf_sa2_0_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'fig');
print(['srl1_h2gf_sa2_0_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'-dtiff');


%%sa3_0
figure('Color',[1 1 1]); hold on;
box_input = AllInv_srl1_h2gf.sa3_0';
prior_index = 6;
if exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1))>min(box_input(:))
    y1 = min(box_input(:))-0.2;
else
    y1 = exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1))-0.2;
end
if exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1))<max(box_input(:))
    y2 = max(box_input(:))+0.2;
else
    y2 = exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1))+0.2;
end
col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
            1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
            0.8 0.0 0.6];
boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
[i,j]=(max(median(box_input(:,1:length(box_input(1,:))))));
mean_sa3_0= mean2(AllInv_srl1_h2gf.sa3_0);
std_sa3_0 = std2(AllInv_srl1_h2gf.sa3_0);
plot([0 length(h2gf_inf.summary)+1],[mean_sa3_0 mean_sa3_0],'black');
plot([0 length(h2gf_inf.summary)+1],[exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1)) exp(h2gf_inf.hgf.c_prc.priormus(prior_index,1))],'r');
axis([0 length(h2gf_inf.summary)+1 y1 y2])
title({['h2gf sa3 0']; ['(mean: ', num2str(mean_sa3_0),'; std: ', num2str(std_sa3_0),')']});
saveas(gcf,['srl1_h2gf_sa3_0_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'fig');
print(['srl1_h2gf_sa3_0_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'-dtiff');

%%ze
figure('Color',[1 1 1]); hold on;
box_input = AllInv_srl1_h2gf.ze';
prior_index = 1;
if exp(h2gf_inf.hgf.c_obs.priormus(prior_index,1))>min(box_input(:))
    y1 = min(box_input(:))-0.2;
else
    y1 = exp(h2gf_inf.hgf.c_obs.priormus(prior_index,1))-0.2;
end
if exp(h2gf_inf.hgf.c_obs.priormus(prior_index,1))<max(box_input(:))
    y2 = max(box_input(:))+0.2;
else
    y2 = exp(h2gf_inf.hgf.c_obs.priormus(prior_index,1))+0.2;
end
col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
            1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
            0.8 0.0 0.6];
boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
[i,j]=(max(median(box_input(:,1:length(box_input(1,:))))));
mean_ze= mean2(AllInv_srl1_h2gf.ze);
std_ze = std2(AllInv_srl1_h2gf.ze);
plot([0 length(h2gf_inf.summary)+1],[mean_ze mean_ze],'black');
plot([0 length(h2gf_inf.summary)+1],[exp(h2gf_inf.hgf.c_obs.priormus(prior_index,1)) exp(h2gf_inf.hgf.c_obs.priormus(prior_index,1))],'r');
axis([0 length(h2gf_inf.summary)+1 y1 y2])
title({['h2gf ze']; ['(mean: ', num2str(mean_ze),'; std: ', num2str(std_ze),')']});
saveas(gcf,['srl1_h2gf_ze_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'fig');
print(['srl1_h2gf_ze_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'-dtiff');

% close all;
end


