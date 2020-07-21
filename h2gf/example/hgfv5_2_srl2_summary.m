% Summary of HGF results, using the data: PRSSI, EEG, short version of SRL
% (srl2)
% =========================================================================
% hgfv5_2_srl2_summary
% =========================================================================

function hgfv5_2_srl2_summary(config_file)

addpath(genpath('/cluster/project/tnu/igsandra/tapas/'));

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

%% define where to store results:
f = mfilename('fullpath');

[tdir, ~, ~] = fileparts(f);

maskResFolder = ([tdir,'/results/hgf/']);
cd ([maskResFolder,configtype]);

%% We load the behavioural srl2 data
disp('============================');
disp(['This is  model (',mat2str(configtype),')']);
disp('============================');

if config_file == 13
    hgf_est_srl2 = load(['hgf_rw_est_srl2_',configtype,'.mat']);
    hgf_est_srl2 = hgf_est_srl2.hgf_est_srl2;
    for s = 1:length(hgf_est_srl2)
        %% estimated parameters
        hgf_est_srl2_summary.v_0(s,1) = hgf_est_srl2(s).p_prc.v_0;
        hgf_est_srl2_summary.al(s,1) = hgf_est_srl2(s).p_prc.al;
        hgf_est_srl2_summary.ze(s,1) = hgf_est_srl2(s).p_obs.ze;
        hgf_est_srl2_summary.LME(s,1) = hgf_est_srl2(s).optim.LME;
        hgf_est_srl2_summary.accu(s,1) = hgf_est_srl2(s).optim.accu;
        hgf_est_srl2_summary.comp(s,1) = hgf_est_srl2(s).optim.comp;
        
        save(['hgf_rw_est_srl2_summary',configtype,'.mat'],'hgf_est_srl2_summary');
    end
    % plot results
    figure('Color',[1 1 1],'OuterPosition',[10 10 1300 700]);hold on;
    %%v_0
    hold on;
    subplot(1,6,1); hold on;
    box_input = hgf_est_srl2_summary.v_0;
    prior_index = 1;
    col_input = [0.4 0.0 0.6];
    boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
    mean_v_0= mean2(hgf_est_srl2_summary.v_0);
    std_v_0 = std2(hgf_est_srl2_summary.v_0);
    plot([0 2],[mean_v_0 mean_v_0],'black');
    if tapas_sgm(hgf_est_srl2(1).c_prc.priormus(1,prior_index),1)>min(box_input(:))
        y1 = min(box_input(:))-0.2;
    else
        y1 = tapas_sgm(hgf_est_srl2(1).c_prc.priormus(1,prior_index),1)-0.2;
    end
    if tapas_sgm(hgf_est_srl2(1).c_prc.priormus(1,prior_index),1)<max(box_input(:))
        y2 = max(box_input(:))+0.2;
    else
        y2 = tapas_sgm(hgf_est_srl2(1).c_prc.priormus(1,prior_index),1)+0.2;
    end
    plot([0 2],[tapas_sgm(hgf_est_srl2(1).c_prc.priormus(1,prior_index),1) tapas_sgm(hgf_est_srl2(1).c_prc.priormus(1,prior_index),1)],'r');
    axis([0 2 y1 y2])
    title({['h2gf v 0']; ['(mean: ', num2str(round(mean_v_0,1)),'; std: ', num2str(round(std_v_0,1)),')']});
    
    %%al
    hold on;
    subplot(1,6,2); hold on;
    box_input = hgf_est_srl2_summary.al;
    prior_index = 2;
    col_input = [0.4 0.0 0.6];
    boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
    mean_al= mean2(hgf_est_srl2_summary.al);
    std_al = std2(hgf_est_srl2_summary.al);
    plot([0 2],[mean_al mean_al],'black');
    if tapas_sgm(hgf_est_srl2(1).c_prc.priormus(1,prior_index),1)>min(box_input(:))
        y1 = min(box_input(:))-0.2;
    else
        y1 = tapas_sgm(hgf_est_srl2(1).c_prc.priormus(1,prior_index),1)-0.2;
    end
    if tapas_sgm(hgf_est_srl2(1).c_prc.priormus(1,prior_index),1)<max(box_input(:))
        y2 = max(box_input(:))+0.2;
    else
        y2 = tapas_sgm(hgf_est_srl2(1).c_prc.priormus(1,prior_index),1)+0.2;
    end
    plot([0 2],[tapas_sgm(hgf_est_srl2(1).c_prc.priormus(1,prior_index),1) tapas_sgm(hgf_est_srl2(1).c_prc.priormus(1,prior_index),1)],'r');
    axis([0 2 y1 y2])
    title({['h2gf alpha']; ['(mean: ', num2str(round(mean_al,1)),'; std: ', num2str(round(std_al,1)),')']});

    %%ze
    hold on;
    subplot(1,6,3); hold on;
    box_input = hgf_est_srl2_summary.ze;
    col_input = [0.4 0.0 0.6];
    boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
    mean_ze= mean2(hgf_est_srl2_summary.ze);
    std_ze = std2(hgf_est_srl2_summary.ze);
    plot([0 2],[mean_ze mean_ze],'black');
    if exp(hgf_est_srl2(1).c_obs.priormus(1,1))>min(box_input(:))
        y1 = min(box_input(:))-0.2;
    else
        y1 = exp(hgf_est_srl2(1).c_obs.priormus(1,1))-0.2;
    end
    if exp(hgf_est_srl2(1).c_obs.priormus(1,1))<max(box_input(:))
        y2 = max(box_input(:))+0.2;
    else
        y2 = exp(hgf_est_srl2(1).c_obs.priormus(1,1))+0.2;
    end
    plot([0 2],[exp(hgf_est_srl2(1).c_obs.priormus(1,1)) exp(hgf_est_srl2(1).c_obs.priormus(1,1))],'r');
    axis([0 2 y1 y2])
    title({['h2gf ze']; ['(mean: ', num2str(mean_ze),'; std: ', num2str(std_ze),')']});
    
    %%LME
    box_input = hgf_est_srl2_summary.LME;
    subplot(1,6,4); hold on;
    % col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
    %             1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
    %             0.8 0.0 0.6];
    col_input = [0.4 0.0 0.6];
    boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
    mean_LME= mean2(hgf_est_srl2_summary.LME);
    std_LME = std2(hgf_est_srl2_summary.LME);
    plot([0 2],[mean_LME mean_LME],'black');
    title({['hgf LME']; ['(mean: ', num2str(round(mean_LME,1)),'; std: ', num2str(round(std_LME,1)),')']});
    
    %%accu
    box_input = hgf_est_srl2_summary.accu;
    subplot(1,6,5); hold on;
    % col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
    %             1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
    %             0.8 0.0 0.6];
    col_input = [0.4 0.0 0.6];
    boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
    mean_accu= mean2(hgf_est_srl2_summary.accu);
    std_accu = std2(hgf_est_srl2_summary.accu);
    plot([0 2],[mean_accu mean_accu],'black');
    title({['hgf accuracy']; ['(mean: ', num2str(round(mean_accu,1)),'; std: ', num2str(round(std_accu,1)),')']});
    
    %%comp
    box_input = hgf_est_srl2_summary.comp;
    subplot(1,6,6); hold on;
    % col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
    %             1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
    %             0.8 0.0 0.6];
    col_input = [0.4 0.0 0.6];
    boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
    mean_comp= mean2(hgf_est_srl2_summary.comp);
    std_comp = std2(hgf_est_srl2_summary.comp);
    plot([0 2],[mean_comp mean_comp],'black');
    title({['hgf complexity']; ['(mean: ', num2str(round(mean_comp,1)),'; std: ', num2str(round(std_comp,1)),')']});
    
    %save plot
    cd(maskResFolder);
    saveas(gcf,['hgf_est_srl2_boxplot_',configtype],'fig');
    print(['hgf_est_srl2_boxplot',configtype],'-dtiff');
else
    hgf_est_srl2 = load(['hgf_3l_est_srl2_',configtype,'.mat']);
    hgf_est_srl2 = hgf_est_srl2.hgf_est_srl2;
    for s = 1:length(hgf_est_srl2)
        %% estimated parameters
        hgf_est_srl2_summary.mu2_0(s,1) = hgf_est_srl2(s).p_prc.mu_0(1,2);
        hgf_est_srl2_summary.mu3_0(s,1) = hgf_est_srl2(s).p_prc.mu_0(1,3);
        hgf_est_srl2_summary.sa2_0(s,1) = hgf_est_srl2(s).p_prc.sa_0(1,2);
        hgf_est_srl2_summary.sa3_0(s,1) = hgf_est_srl2(s).p_prc.sa_0(1,3);
        hgf_est_srl2_summary.ka(s,1) = hgf_est_srl2(s).p_prc.ka(1,2);
        hgf_est_srl2_summary.om2(s,1) = hgf_est_srl2(s).p_prc.om(1,2);
        hgf_est_srl2_summary.om3(s,1) = hgf_est_srl2(s).p_prc.om(1,3);
        hgf_est_srl2_summary.ze(s,1) = hgf_est_srl2(s).p_obs.ze;
        hgf_est_srl2_summary.LME(s,1) = hgf_est_srl2(s).optim.LME;
        hgf_est_srl2_summary.accu(s,1) = hgf_est_srl2(s).optim.accu;
        hgf_est_srl2_summary.comp(s,1) = hgf_est_srl2(s).optim.comp;
        
        save(['hgf_3l_est_srl2_summary',configtype,'.mat'],'hgf_est_srl2_summary');
    end
    
    % plot results
    figure('Color',[1 1 1]); hold on;
    %%kappa
    hold on;
    subplot(2,6,1); hold on;
    box_input = hgf_est_srl2_summary.ka;
    prior_index = 11;
    col_input = [0.4 0.0 0.6];
    boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
    mean_ka= mean2(hgf_est_srl2_summary.ka);
    std_ka = std2(hgf_est_srl2_summary.ka);
    plot([0 2],[mean_ka mean_ka],'black');
    if exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index))>min(box_input(:))
        y1 = min(box_input(:))-0.2;
    else
        y1 = exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index))-0.2;
    end
    if exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index))<max(box_input(:))
        y2 = max(box_input(:))+0.2;
    else
        y2 = exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index))+0.2;
    end
    plot([0 2],[exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index)) exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index))],'r');
    axis([0 2 y1 y2])
    title({['hgf ka']; ['(mean: ', num2str(round(mean_ka,1)),'; std: ', num2str(round(std_ka,1)),')']});
    
    %%omega2
    hold on;
    subplot(2,6,2); hold on;
    box_input = hgf_est_srl2_summary.om2;
    prior_index = 13;
    col_input = [0.4 0.0 0.6];
    boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
    mean_om2= mean2(hgf_est_srl2_summary.om2);
    std_om2 = std2(hgf_est_srl2_summary.om2);
    plot([0 2],[mean_om2 mean_om2],'black');
    if hgf_est_srl2(1).c_prc.priormus(1,prior_index)>min(box_input(:))
        y1 = min(box_input(:))-0.2;
    else
        y1 = hgf_est_srl2(1).c_prc.priormus(1,prior_index)-0.2;
    end
    if hgf_est_srl2(1).c_prc.priormus(1,prior_index)<max(box_input(:))
        y2 = max(box_input(:))+0.2;
    else
        y2 = hgf_est_srl2(1).c_prc.priormus(1,prior_index)+0.2;
    end
    plot([0 2],[hgf_est_srl2(1).c_prc.priormus(1,prior_index) hgf_est_srl2(1).c_prc.priormus(1,prior_index)],'r');
    axis([0 2 y1 y2])
    title({['hgf om2']; ['(mean: ', num2str(round(mean_om2,1)),'; std: ', num2str(round(std_om2,1)),')']});
    
    %%omega3
    hold on;
    subplot(2,6,3); hold on;
    box_input = hgf_est_srl2_summary.om3;
    prior_index = 14;
    col_input = [0.4 0.0 0.6];
    boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
    mean_om3= mean2(hgf_est_srl2_summary.om3);
    std_om3 = std2(hgf_est_srl2_summary.om3);
    plot([0 2],[mean_om3 mean_om3],'black');
    if hgf_est_srl2(1).c_prc.priormus(1,prior_index)>min(box_input(:))
        y1 = min(box_input(:))-0.2;
    else
        y1 = hgf_est_srl2(1).c_prc.priormus(1,prior_index)-0.2;
    end
    if hgf_est_srl2(1).c_prc.priormus(1,prior_index)<max(box_input(:))
        y2 = max(box_input(:))+0.2;
    else
        y2 = hgf_est_srl2(1).c_prc.priormus(1,prior_index)+0.2;
    end
    plot([0 2],[hgf_est_srl2(1).c_prc.priormus(1,prior_index) hgf_est_srl2(1).c_prc.priormus(1,prior_index)],'r');
    axis([0 2 y1 y2])
    title({['hgf om3']; ['(mean: ', num2str(round(mean_om3,1)),'; std: ', num2str(round(std_om3,1)),')']});
    
    %%mu2_0
    hold on;
    subplot(2,6,4); hold on;
    box_input = hgf_est_srl2_summary.mu2_0;
    prior_index = 2;
    col_input = [0.4 0.0 0.6];
    boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
    mean_mu2_0= mean2(hgf_est_srl2_summary.mu2_0);
    std_mu2_0 = std2(hgf_est_srl2_summary.mu2_0);
    plot([0 2],[mean_mu2_0 mean_mu2_0],'black');
    if hgf_est_srl2(1).c_prc.priormus(1,prior_index)>min(box_input(:))
        y1 = min(box_input(:))-0.2;
    else
        y1 = hgf_est_srl2(1).c_prc.priormus(1,prior_index)-0.2;
    end
    if hgf_est_srl2(1).c_prc.priormus(1,prior_index)<max(box_input(:))
        y2 = max(box_input(:))+0.2;
    else
        y2 = hgf_est_srl2(1).c_prc.priormus(1,prior_index)+0.2;
    end
    plot([0 2],[hgf_est_srl2(1).c_prc.priormus(1,prior_index) hgf_est_srl2(1).c_prc.priormus(1,prior_index)],'r');
    axis([0 2 y1 y2])
    title({['hgf mu2 0']; ['(mean: ', num2str(round(mean_mu2_0,1)),'; std: ', num2str(round(std_mu2_0,1)),')']});
    
    
    %%mu3_0
    hold on;
    subplot(2,6,5); hold on;
    box_input = hgf_est_srl2_summary.mu3_0;
    prior_index = 3;
    col_input = [0.4 0.0 0.6];
    boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
    mean_mu3_0= mean2(hgf_est_srl2_summary.mu3_0);
    std_mu3_0 = std2(hgf_est_srl2_summary.mu3_0);
    plot([0 2],[mean_mu3_0 mean_mu3_0],'black');
    if hgf_est_srl2(1).c_prc.priormus(1,prior_index)>min(box_input(:))
        y1 = min(box_input(:))-0.2;
    else
        y1 = hgf_est_srl2(1).c_prc.priormus(1,prior_index)-0.2;
    end
    if hgf_est_srl2(1).c_prc.priormus(1,prior_index)<max(box_input(:))
        y2 = max(box_input(:))+0.2;
    else
        y2 = hgf_est_srl2(1).c_prc.priormus(1,prior_index)+0.2;
    end
    plot([0 2],[hgf_est_srl2(1).c_prc.priormus(1,prior_index) hgf_est_srl2(1).c_prc.priormus(1,prior_index)],'r');
    axis([0 2 y1 y2])
    title({['hgf mu3 0']; ['(mean: ', num2str(round(mean_mu3_0,1)),'; std: ', num2str(round(std_mu3_0,1)),')']});
    
    
    %%sa2_0
    hold on;
    subplot(2,6,6); hold on;
    box_input = hgf_est_srl2_summary.sa2_0;
    prior_index = 5;
    col_input = [0.4 0.0 0.6];
    boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
    mean_sa2_0= mean2(hgf_est_srl2_summary.sa2_0);
    std_sa2_0 = std2(hgf_est_srl2_summary.sa2_0);
    plot([0 2],[mean_sa2_0 mean_sa2_0],'black');
    if exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index))>min(box_input(:))
        y1 = min(box_input(:))-0.2;
    else
        y1 = exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index))-0.2;
    end
    if exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index))<max(box_input(:))
        y2 = max(box_input(:))+0.2;
    else
        y2 = exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index))+0.2;
    end
    plot([0 2],[exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index)) exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index))],'r');
    axis([0 2 y1 y2])
    title({['hgf sa2 0']; ['(mean: ', num2str(round(mean_sa2_0,1)),'; std: ', num2str(round(std_sa2_0,1)),')']});
    
    
    %%sa3_0
    hold on;
    subplot(2,6,7); hold on;
    box_input = hgf_est_srl2_summary.sa3_0;
    prior_index = 6;
    col_input = [0.4 0.0 0.6];
    boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
    mean_sa3_0= mean2(hgf_est_srl2_summary.sa3_0);
    std_sa3_0 = std2(hgf_est_srl2_summary.sa3_0);
    plot([0 2],[mean_sa3_0 mean_sa3_0],'black');
    if exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index))>min(box_input(:))
        y1 = min(box_input(:))-0.2;
    else
        y1 = exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index))-0.2;
    end
    if exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index))<max(box_input(:))
        y2 = max(box_input(:))+0.2;
    else
        y2 = exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index))+0.2;
    end
    plot([0 2],[exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index)) exp(hgf_est_srl2(1).c_prc.priormus(1,prior_index))],'r');
    axis([0 2 y1 y2])
    title({['hgf sa3 0']; ['(mean: ', num2str(round(mean_sa3_0,1)),'; std: ', num2str(round(std_sa3_0,1)),')']});
    
    %%ze
    hold on;
    subplot(2,6,8); hold on;
    box_input = hgf_est_srl2_summary.ze;
    col_input = [0.4 0.0 0.6];
    boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
    mean_ze= mean2(hgf_est_srl2_summary.ze);
    std_ze = std2(hgf_est_srl2_summary.ze);
    plot([0 2],[mean_ze mean_ze],'black');
    if exp(hgf_est_srl2(1).c_obs.priormus(1,1))>min(box_input(:))
        y1 = min(box_input(:))-0.2;
    else
        y1 = exp(hgf_est_srl2(1).c_obs.priormus(1,1))-0.2;
    end
    if exp(hgf_est_srl2(1).c_obs.priormus(1,1))<max(box_input(:))
        y2 = max(box_input(:))+0.2;
    else
        y2 = exp(hgf_est_srl2(1).c_obs.priormus(1,1))+0.2;
    end
    plot([0 2],[exp(hgf_est_srl2(1).c_obs.priormus(1,1)) exp(hgf_est_srl2(1).c_obs.priormus(1,1))],'r');
    axis([0 2 y1 y2])
    title({['hgf ze']; ['(mean: ', num2str(mean_ze),'; std: ', num2str(std_ze),')']});
    
    %%LME
    box_input = hgf_est_srl2_summary.LME;
    subplot(2,6,9); hold on;
    % col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
    %             1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
    %             0.8 0.0 0.6];
    col_input = [0.4 0.0 0.6];
    boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
    mean_LME= mean2(hgf_est_srl2_summary.LME);
    std_LME = std2(hgf_est_srl2_summary.LME);
    plot([0 2],[mean_LME mean_LME],'black');
    title({['hgf LME']; ['(mean: ', num2str(round(mean_LME,1)),'; std: ', num2str(round(std_LME,1)),')']});
    
    %%accu
    box_input = hgf_est_srl2_summary.accu;
    subplot(2,6,10); hold on;
    % col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
    %             1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
    %             0.8 0.0 0.6];
    col_input = [0.4 0.0 0.6];
    boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
    mean_accu= mean2(hgf_est_srl2_summary.accu);
    std_accu = std2(hgf_est_srl2_summary.accu);
    plot([0 2],[mean_accu mean_accu],'black');
    title({['hgf accuracy']; ['(mean: ', num2str(round(mean_accu,1)),'; std: ', num2str(round(std_accu,1)),')']});
    
    %%comp
    box_input = hgf_est_srl2_summary.comp;
    subplot(2,6,11); hold on;
    % col_input = [0.4 0.0 0.6; 0.4 0.2 0.6; 0.4 0.4 0.6; 0.4 0.6 0.6; 0.4 0.8 0.6; 0.4 1.0 0.6; ...
    %             1.0 1.0 0.6; 1.0 0.8 0.6; 1.0 0.6 0.6; 1.0 0.4 0.6; 1.0 0.2 0.6; 0.8 0.4 0.6; ...
    %             0.8 0.0 0.6];
    col_input = [0.4 0.0 0.6];
    boxplot(box_input,'colors',col_input, 'Plotstyle','compact'); hold on;
    mean_comp= mean2(hgf_est_srl2_summary.comp);
    std_comp = std2(hgf_est_srl2_summary.comp);
    plot([0 2],[mean_comp mean_comp],'black');
    title({['hgf complexity']; ['(mean: ', num2str(round(mean_comp,1)),'; std: ', num2str(round(std_comp,1)),')']});
    
    suptitle(['SRL2; ', configtype]);
    %save plot
    cd(maskResFolder);
    saveas(gcf,[configtype,'_hgf_est_srl2_boxplot'],'fig');
    print([configtype,'_hgf_est_srl2_boxplot'],'-dtiff');
    print([configtype,'_hgf_est_srl2_boxplot'],'-dpdf');
    
end

