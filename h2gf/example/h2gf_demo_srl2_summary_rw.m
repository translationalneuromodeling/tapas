% h2gf demo using the data: PRSSI, EEG, short version of SRL (srl2)
%
% plot boxplots for all parameters and LME
% =========================================================================
% h2gf_demo_srl2_summary(4000,1,1)
% =========================================================================

function h2gf_demo_srl2_summary_rw(NrIter,spec_eta)

addpath(genpath('/cluster/project/tnu/igsandra/tapas/'));

%% specify eta_label
eta_label = num2str(spec_eta);

disp('config file: rw');
disp('**************************************');

%% define where results have been stored:
f = mfilename('fullpath');

[tdir, ~, ~] = fileparts(f);

maskResFolder = ([tdir,'/results/',configtype,'/eta', eta_label,'/', num2str(NrIter)]);
cd(maskResFolder);

for m = 1:12 %length(listFiles)
    clear h2gf_inf;
    % Find data:
    disp(['This is inference Nr.: (',mat2str(m),')']);

    h2gf_inf = load([maskResFolder, '/h2gf_rw_est_srl2_',configtype,'_eta',eta_label,'_', num2str(NrIter),'_',num2str(m),'.mat']);
    
    for s = 1:length(h2gf_inf.summary)
        
        %% estimated parameters
        srl2_estpar_h2gf.param.p_prc = h2gf_inf.summary(s).p_prc;
        srl2_estpar_h2gf.param.p_obs = h2gf_inf.summary(s).p_obs;
        srl2_estpar_h2gf.param.LME = h2gf_inf.summary(s).optim.LME;
        
        srl2_estpar_h2gf.param.v_0 = srl2_estpar_h2gf.param.p_prc.v_0;
        srl2_estpar_h2gf.param.al = srl2_estpar_h2gf.param.p_prc.al;
                
        srl2_estpar_h2gf.param.ze = srl2_estpar_h2gf.param.p_obs.ze;
        
        % Structure stats for all        
        AllInv_srl2_h2gf.LME(s,m)  = srl2_estpar_h2gf.param.LME;
        
        AllInv_srl2_h2gf.v_0(s,m)  = srl2_estpar_h2gf.param.v_0;
        AllInv_srl2_h2gf.al(s,m) = srl2_estpar_h2gf.param.al;
        
        AllInv_srl2_h2gf.ze(s,m) = srl2_estpar_h2gf.param.ze;
        
        save (['AllInv_srl2_h2gf_rw_eta',eta_label,'_', num2str(NrIter),'.mat'], '-struct','AllInv_srl2_h2gf');
        clear srl2_estpar_h2gf;
        
    end
    
end

%%lme
figure('Color',[1 1 1]); hold on;
box_input = AllInv_srl2_h2gf.LME';
figure('Color',[1 1 1]); hold on;
col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
            1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
            0.8 0.0 0.6];
boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
[i,j]=(max(median(box_input(:,1:length(box_input(1,:))))));
mean_LME= mean2(AllInv_srl2_h2gf.LME);
std_LME = std2(AllInv_srl2_h2gf.LME);
plot([0 length(h2gf_inf.summary)+1],[mean_LME mean_LME],'black');
title({['h2gf LME']; ['(mean: ', num2str(mean_LME),'; std: ', num2str(std_LME),')']});
saveas(gcf,['srl2_h2gf_LME_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'fig');
print(['srl2_h2gf_LME_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'-dtiff');


%%v_0
figure('Color',[1 1 1]); hold on;
box_input = AllInv_srl2_h2gf.v_0';
prior_index = 1;
figure('Color',[1 1 1]); hold on;
col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
            1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
            0.8 0.0 0.6];
boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
[i,j]=(max(median(box_input(:,1:length(box_input(1,:))))));
mean_v_0= mean2(AllInv_srl2_h2gf.v_0);
std_v_0 = std2(AllInv_srl2_h2gf.v_0);
if tapas_sgm(h2gf_inf.hgf.c_prc.priormus(prior_index,1),1)>min(box_input(:))
    y1 = min(box_input(:))-0.2;
else
    y1 = tapas_sgm(h2gf_inf.hgf.c_prc.priormus(prior_index,1),1)-0.2;
end
if tapas_sgm(h2gf_inf.hgf.c_prc.priormus(prior_index,1),1)<max(box_input(:))
    y2 = max(box_input(:))+0.2;
else
    y2 = tapas_sgm(h2gf_inf.hgf.c_prc.priormus(prior_index,1),1)+0.2;
end
plot([0 length(h2gf_inf.summary)+1],[mean_v_0 mean_v_0],'black'); hold on;
plot([0 length(h2gf_inf.summary)+1],[tapas_sgm(h2gf_inf.hgf.c_prc.priormus(prior_index,1),1) tapas_sgm(h2gf_inf.hgf.c_prc.priormus(prior_index,1),1)],'r');
axis([0 length(h2gf_inf.summary)+1 y1 y2])
title({['h2gf v 0']; ['(mean: ', num2str(mean_v_0),'; std: ', num2str(std_v_0),')']});
saveas(gcf,['srl2_h2gf_v_0_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'fig');
print(['srl2_h2gf_v_0_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'-dtiff');


%%al
figure('Color',[1 1 1]); hold on;
box_input = AllInv_srl2_h2gf.al';
prior_index = 2;
col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
            1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
            0.8 0.0 0.6];
boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
[i,j]=(max(median(box_input(:,1:length(box_input(1,:))))));
mean_al= mean2(AllInv_srl2_h2gf.al);
std_al = std2(AllInv_srl2_h2gf.al);

plot([0 length(h2gf_inf.summary)+1],[mean_al mean_al],'black'); hold on;
plot([0 length(h2gf_inf.summary)+1],[tapas_sgm(h2gf_inf.hgf.c_prc.priormus(prior_index,1),1) tapas_sgm(h2gf_inf.hgf.c_prc.priormus(prior_index,1),1)],'r');

title({['h2gf alpha']; ['(mean: ', num2str(mean_al),'; std: ', num2str(std_al),')']});
saveas(gcf,['srl2_h2gf_al_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'fig');
print(['srl2_h2gf_al_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'-dtiff');

%%ze
figure('Color',[1 1 1]); hold on;
box_input = AllInv_srl2_h2gf.ze';
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
mean_ze= mean2(AllInv_srl2_h2gf.ze);
std_ze = std2(AllInv_srl2_h2gf.ze);
plot([0 length(h2gf_inf.summary)+1],[mean_ze mean_ze],'black');
plot([0 length(h2gf_inf.summary)+1],[exp(h2gf_inf.hgf.c_obs.priormus(prior_index,1)) exp(h2gf_inf.hgf.c_obs.priormus(prior_index,1))],'r');
axis([0 length(h2gf_inf.summary)+1 y1 y2])
title({['h2gf ze']; ['(mean: ', num2str(mean_ze),'; std: ', num2str(std_ze),')']});
saveas(gcf,['srl2_h2gf_ze_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'fig');
print(['srl2_h2gf_ze_boxplot_',configtype,'_eta',eta_label,'_', num2str(NrIter)],'-dtiff');

end


