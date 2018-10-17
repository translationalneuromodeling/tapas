% h2gf demo using the data: PRSSI, EEG, short version of SRL (srl2)
%
% plot h2gf results
% =========================================================================
% h2gf_demo_srl2_plot_data(4000,1,1) 
% =========================================================================
***not ready yet***
function h2gf_demo_srl2_plot_data(NrIter,spec_eta,config_file)

addpath(genpath('/cluster/project/tnu/igsandra/tapas/'));

disp(['Nr samples stored:', num2str(NrIter)]);
disp('**************************************');
disp(['eta set to:', num2str(spec_eta)]);
disp('**************************************');

% options = prssi_set_analysis_options_srl2;
clear SRL

maskModel = {'HGF_1_fixom_v5_1'};

disp(['This is hgfToolBox_v5.1 srl EEG study 2:', maskModel]);% Go through scans

%% specify eta:
eta_label = num2str(spec_eta);
if spec_eta == 1
    eta_v = spec_eta;
elseif spec_eta == 2
    eta_v = 10;
elseif spec_eta == 3
    eta_v = 20;
elseif spec_eta == 4
    eta_v = 40;
elseif spec_eta == 5
    eta_v = [1 1 1 1 1 1 1 1 1 1 1 1 1 5 1]';
    % mu1_0, mu2_0, mu3_0, sa1_0, sa2_0, sa3_0,
    % rho1, rho2, rho3, ka1, ka2, om1, om2, om3
    % ze
elseif spec_eta == 6
    eta_v = [1 1 1 1 1 1 1 1 1 1 1 1 1 10 1]';
    % mu1_0, mu2_0, mu3_0, sa1_0, sa2_0, sa3_0,
    % rho1, rho2, rho3, ka1, ka2, om1, om2, om3
    % ze
end

%% specify which configuration of the binary hgf has been used
if config_file == 1
    config = tapas_hgf_binary_config_estka2_new();
    configtype = 'estka2';
elseif config_file == 2
    config = tapas_hgf_binary_config_estka2mu2_new();
    configtype = 'estka2mu2';
elseif config_file == 3
    config = tapas_hgf_binary_config_estka2mu3_new();
    configtype = 'estka2mu3';
elseif config_file == 4
    config = tapas_hgf_binary_config_estka2om3_new();
    configtype = 'estka2om3';
elseif config_file == 5
    config = tapas_hgf_binary_config_estka2sa2_new();
    configtype = 'estka2sa2';
elseif config_file == 6
    config = tapas_hgf_binary_config_estka2sa3_new();
    configtype = 'estka2sa3';
elseif config_file == 7
    config = tapas_hgf_binary_config_estom2_new();
    configtype = 'estom2';
elseif config_file == 8
    config = tapas_hgf_binary_config_estom2mu2_new();
    configtype = 'estom2mu2';
elseif config_file == 9
    config = tapas_hgf_binary_config_estom2mu3_new();
    configtype = 'estom2mu3';
elseif config_file == 10
    config = tapas_hgf_binary_config_estom2om3_new();
    configtype = 'estom2om3';
elseif config_file == 11
    config = tapas_hgf_binary_config_estom2sa2_new();
    configtype = 'estom2sa2';
elseif config_file == 12
    config = tapas_hgf_binary_config_estom2sa3_new();
    configtype = 'estom2sa3';
end

disp(['config file:', configtype]);
disp('**************************************');

%% define where results have been stored:
f = mfilename('fullpath');

[tdir, ~, ~] = fileparts(f);

maskResFolder = ([tdir,'/results/',configtype,'/eta', eta_label,'/', num2str(NrIter)]);

h2gf_est = load([maskResFolder, 'h2gf_3l_est_srl2_',configtype,'_eta',eta_label,'_', num2str(NrIter),'_',num2str(m),'.mat']);

subjindex = 0;
for idCell = options.subjectIDs
    subjindex = subjindex+1;
    disp('_________________________________________')
    id = char(idCell);
    disp('_________________________________________');
    details = prssi_subjects_srl2(id);
    id
    if id == '0362'
        continue
    end
    cd ([details.behavrootresults,maskModel{1}]);
    tapas_h2gf_binary_plotTraj(h2gf_est.summary(subjindex).traj, ...
        h2gf_est.data(subjindex).y, h2gf_est.data(subjindex).u, ...
        h2gf_est.summary(subjindex).prc_mean, h2gf_est.summary(subjindex).obs_mean)
    cd ([details.behavrootresults,maskModel{1}]);
    print(['srl_re_h2gf_3l_fixom_eta',eta_label,'_', num2str(NrIter)],'-dtiff');
    %             saveas (gcf,'srl_re','fig');
    % copy srl.tif plots to srl_all folder
    cd (maskResFolder);
    mkdir (maskModel{1});
    cd ([details.behavrootresults,maskModel{1}]);
    copyfile (['srl_re_h2gf_3l_fixom_eta', eta_label,'_', num2str(NrIter),'.tif'], [maskResFolder, maskModel{1},'/srl_re_h2gf_3l_fixom_eta', eta_label,'_', num2str(NrIter), details.subjname,'.tif']);
    cd(options.workdir);
    delete(findall(0,'Type','figure'));
end
end