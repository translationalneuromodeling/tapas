% h2gf demo using the data: PRSSI, EEG, long version of SRL (srl1)
%
% plot h2gf results
% =========================================================================
% h2gf_demo_srl1_plot_data(4000,1,1)
% =========================================================================

function h2gf_demo_srl1_plot_data(NrIter,spec_eta,config_file,m)

addpath(genpath('/cluster/project/tnu/igsandra/tapas/'));

disp(['Nr samples stored:', num2str(NrIter)]);
disp('**************************************');
disp(['eta set to:', num2str(spec_eta)]);
disp('**************************************');

maskRep = ['/traj_',num2str(m)];

disp(['This is SRL EEG study 1']);% Go through scans

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
elseif config_file == 13
    configtype = 'estrw';
end

disp(['config file:', configtype]);
disp('**************************************');

%% define where results have been stored:
f = mfilename('fullpath');

[tdir, ~, ~] = fileparts(f);

maskResFolder = ([tdir,'/results/',configtype,'/eta', eta_label,'/', num2str(NrIter)]);
maskTrajFolder = fullfile([maskResFolder, maskRep]);
mkdir(maskTrajFolder);

if config_file == 13
    h2gf_est = load([maskResFolder, '/h2gf_rw_est_srl1_',configtype,'_eta',eta_label,'_', num2str(NrIter),'_',num2str(m),'.mat']);
else
    h2gf_est = load([maskResFolder, '/h2gf_3l_est_srl1_',configtype,'_eta',eta_label,'_', num2str(NrIter),'_',num2str(m),'.mat']);
end

subjindex = 0;
for idCell = 1:length(h2gf_est.summary)
    subjindex = subjindex+1;
    disp('_________________________________________')
    disp(['subject nr.: ',num2str(subjindex)]);
    disp('_________________________________________');
    
    cd (maskTrajFolder);
    if config_file == 13
        tapas_rw_binary_plotTraj(h2gf_est.summary(subjindex))
        print(['srl1_re_h2gf_rw_',configtype,'_eta',eta_label,'_', num2str(NrIter),'_subjnr_',num2str(subjindex)],'-dtiff');
    else
        tapas_hgf_binary_plotTraj(h2gf_est.summary(subjindex))
        print(['srl1_re_h2gf_3l_',configtype,'_eta',eta_label,'_', num2str(NrIter),'_subjnr_',num2str(subjindex)],'-dtiff');
    end
    
    delete(findall(0,'Type','figure'));
end
end