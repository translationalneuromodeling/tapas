% h2gf demo using the data: PRSSI, EEG, long version of SRL (srl1)
%
% plot h2gf results
% =========================================================================
% h2gf_demo_srl1_plot_inference(1,3,1)
% =========================================================================
function h2gf_demo_srl1_plot_inference(spec_eta,config_file,m)

addpath(genpath('/cluster/project/tnu/igsandra/tapas/'));

disp(['Nr configuration file:', num2str(config_file)]);
disp('**************************************');
disp(['eta set to:', num2str(spec_eta)]);
disp('**************************************');

maskRep = ['/plotinf_',num2str(m)];


disp(['This is SRL EEG study 1']);% Go through scans
%% specify Nr of interations
NrIter = [1000 3000 4000 5000];

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
    parameter_label = {'ka2'};
    parameter_nr = 11;
elseif config_file == 2
    configtype = 'estka2mu2';
    parameter_label = {'mu2', 'ka2'};
    parameter_nr = [2, 11];
elseif config_file == 3
    configtype = 'estka2mu3';
    parameter_label = {'mu3', 'ka2'};
    parameter_nr = [3, 11];
elseif config_file == 4
    configtype = 'estka2om3';
    parameter_label = {'ka2', 'om3'};
    parameter_nr = [11, 14];
elseif config_file == 5
    configtype = 'estka2sa2';
    parameter_label = {'sa2', 'ka2'};
    parameter_nr = [5, 11];
elseif config_file == 6
    configtype = 'estka2sa3';
    parameter_label = {'sa3', 'ka2'};
    parameter_nr = [6, 11];
elseif config_file == 7
    configtype = 'estom2';
    parameter_label = {'om2'};
    parameter_nr = [13];
elseif config_file == 8
    configtype = 'estom2mu2';
    parameter_label = {'mu2', 'om2'};
    parameter_nr = [2, 13];
elseif config_file == 9
    configtype = 'estom2mu3';
    parameter_label = {'mu3', 'om2'};
    parameter_nr = [3, 13];
elseif config_file == 10
    configtype = 'estom2om3';
    parameter_label = {'om2', 'om3'};
    parameter_nr = [13, 14];
elseif config_file == 11
    configtype = 'estom2sa2';
    parameter_label = {'sa2', 'om2'};
    parameter_nr = [5, 13];
elseif config_file == 12
    configtype = 'estom2sa3';
    parameter_label = {'sa3', 'om2'};
    parameter_nr = [6, 13];
end

disp(['config file:', configtype]);
disp('**************************************');

%% define where results have been stored:
f = mfilename('fullpath');

[tdir, ~, ~] = fileparts(f);

maskResFolder = ([tdir,'/results/',configtype,'/eta', eta_label,'/']);
% maskResFolder = (['D:\PRSSI\h2gf/results/',configtype,'/eta', eta_label,'/']);
maskPlotFolder = fullfile([maskResFolder, maskRep]);
mkdir(maskPlotFolder);

h2gf_est1000 = load([maskResFolder, '1000/h2gf_3l_est_srl1_',configtype,'_eta',eta_label,'_1000_',num2str(m),'.mat']);
h2gf_est3000 = load([maskResFolder, '3000/h2gf_3l_est_srl1_',configtype,'_eta',eta_label,'_3000_',num2str(m),'.mat']);
h2gf_est4000 = load([maskResFolder, '4000/h2gf_3l_est_srl1_',configtype,'_eta',eta_label,'_4000_',num2str(m),'.mat']);
h2gf_est5000 = load([maskResFolder, '5000/h2gf_3l_est_srl1_',configtype,'_eta',eta_label,'_5000_',num2str(m),'.mat']);

subjindex = 0;
for idCell = 1:length(h2gf_est1000.summary)
    subjindex = subjindex+1;
    disp('_________________________________________');
    disp(['subj nr.:', num2str(subjindex)]);
    disp('_________________________________________');
    
    cd (maskPlotFolder);
    % 1 = mu1_0; 4 = sa1_0; 7 = rho1; 10 = ka1; 13 = om2
    % 2 = mu2_0; 5 = sa2_0; 8 = rho2; 11 = ka2; 14 = om3
    % 3 = mu3_0; 6 = sa3_0; 9 = rho3, 12 = om1; 15 = ze
    
    if config_file == 1 %'estka2';
        for i = 1:1000
            param.prc.ka2.s1000(i)  = h2gf_est1000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s1000(i)   = h2gf_est1000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:3000
            param.prc.ka2.s3000(i)  = h2gf_est3000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s3000(i)   = h2gf_est3000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:4000
            param.prc.ka2.s4000(i)  = h2gf_est4000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s4000(i)   = h2gf_est4000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:5000
            param.prc.ka2.s5000(i)  = h2gf_est5000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s5000(i)   = h2gf_est5000.samples_theta{subjindex,i}(15,1);
        end
    elseif config_file == 2 %'estka2mu2';
        for i = 1:1000
            param.prc.mu2.s1000(i) = h2gf_est1000.samples_theta{subjindex,i}(2,1);
            param.prc.ka2.s1000(i)  = h2gf_est1000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s1000(i)   = h2gf_est1000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:3000
            param.prc.mu2.s3000(i) = h2gf_est3000.samples_theta{subjindex,i}(2,1);
            param.prc.ka2.s3000(i)  = h2gf_est3000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s3000(i)   = h2gf_est3000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:4000
            param.prc.mu2.s4000(i) = h2gf_est4000.samples_theta{subjindex,i}(2,1);
            param.prc.ka2.s4000(i)  = h2gf_est4000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s4000(i)   = h2gf_est4000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:5000
            param.prc.mu2.s5000(i) = h2gf_est5000.samples_theta{subjindex,i}(2,1);
            param.prc.ka2.s5000(i)  = h2gf_est5000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s5000(i)   = h2gf_est5000.samples_theta{subjindex,i}(15,1);
        end
    elseif config_file == 3 %'estka2mu3';
        for i = 1:1000
            param.prc.mu3.s1000(i) = h2gf_est1000.samples_theta{subjindex,i}(3,1);
            param.prc.ka2.s1000(i)  = h2gf_est1000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s1000(i)   = h2gf_est1000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:3000
            param.prc.mu3.s3000(i) = h2gf_est3000.samples_theta{subjindex,i}(3,1);
            param.prc.ka2.s3000(i)  = h2gf_est3000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s3000(i)   = h2gf_est3000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:4000
            param.prc.mu3.s4000(i) = h2gf_est4000.samples_theta{subjindex,i}(3,1);
            param.prc.ka2.s4000(i)  = h2gf_est4000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s4000(i)   = h2gf_est4000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:5000
            param.prc.mu3.s5000(i) = h2gf_est5000.samples_theta{subjindex,i}(3,1);
            param.prc.ka2.s5000(i)  = h2gf_est5000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s5000(i)   = h2gf_est5000.samples_theta{subjindex,i}(15,1);
        end
    elseif config_file == 4 %'estka2om3';
        for i = 1:1000
            param.prc.ka2.s1000(i)  = h2gf_est1000.samples_theta{subjindex,i}(11,1);
            param.prc.om3.s1000(i)  = h2gf_est1000.samples_theta{subjindex,i}(14,1);
            param.obs.ze.s1000(i)   = h2gf_est1000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:3000
            param.prc.ka2.s3000(i)  = h2gf_est3000.samples_theta{subjindex,i}(11,1);
            param.prc.om3.s3000(i)  = h2gf_est3000.samples_theta{subjindex,i}(14,1);
            param.obs.ze.s3000(i)   = h2gf_est3000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:4000
            param.prc.ka2.s4000(i)  = h2gf_est4000.samples_theta{subjindex,i}(11,1);
            param.prc.om3.s4000(i)  = h2gf_est4000.samples_theta{subjindex,i}(14,1);
            param.obs.ze.s4000(i)   = h2gf_est4000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:5000
            param.prc.ka2.s5000(i)  = h2gf_est5000.samples_theta{subjindex,i}(11,1);
            param.prc.om3.s5000(i)  = h2gf_est5000.samples_theta{subjindex,i}(14,1);
            param.obs.ze.s5000(i)   = h2gf_est5000.samples_theta{subjindex,i}(15,1);
        end
    elseif config_file == 5 %'estka2sa2';
        for i = 1:1000
            param.prc.sa2.s1000(i) = h2gf_est1000.samples_theta{subjindex,i}(5,1);
            param.prc.ka2.s1000(i)  = h2gf_est1000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s1000(i)   = h2gf_est1000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:3000
            param.prc.sa2.s3000(i) = h2gf_est3000.samples_theta{subjindex,i}(5,1);
            param.prc.ka2.s3000(i)  = h2gf_est3000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s3000(i)   = h2gf_est3000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:4000
            param.prc.sa2.s4000(i) = h2gf_est4000.samples_theta{subjindex,i}(5,1);
            param.prc.ka2.s4000(i)  = h2gf_est4000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s4000(i)   = h2gf_est4000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:5000
            param.prc.sa2.s5000(i) = h2gf_est5000.samples_theta{subjindex,i}(5,1);
            param.prc.ka2.s5000(i)  = h2gf_est5000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s5000(i)   = h2gf_est5000.samples_theta{subjindex,i}(15,1);
        end
    elseif config_file == 6 %'estka2sa3';
        for i = 1:1000
            param.prc.sa3.s1000(i) = h2gf_est1000.samples_theta{subjindex,i}(6,1);
            param.prc.ka2.s1000(i)  = h2gf_est1000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s1000(i)   = h2gf_est1000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:3000
            param.prc.sa3.s3000(i) = h2gf_est3000.samples_theta{subjindex,i}(6,1);
            param.prc.ka2.s3000(i)  = h2gf_est3000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s3000(i)   = h2gf_est3000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:4000
            param.prc.sa3.s4000(i) = h2gf_est4000.samples_theta{subjindex,i}(6,1);
            param.prc.ka2.s4000(i)  = h2gf_est4000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s4000(i)   = h2gf_est4000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:5000
            param.prc.sa3.s5000(i) = h2gf_est5000.samples_theta{subjindex,i}(6,1);
            param.prc.ka2.s5000(i)  = h2gf_est5000.samples_theta{subjindex,i}(11,1);
            param.obs.ze.s5000(i)   = h2gf_est5000.samples_theta{subjindex,i}(15,1);
        end
    elseif config_file == 7 %'estom2';
        for i = 1:1000
            param.prc.om2.s1000(i)  = h2gf_est1000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s1000(i)   = h2gf_est1000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:3000
            param.prc.om2.s3000(i)  = h2gf_est3000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s3000(i)   = h2gf_est3000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:4000
            param.prc.om2.s4000(i)  = h2gf_est4000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s4000(i)   = h2gf_est4000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:5000
            param.prc.om2.s5000(i)  = h2gf_est5000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s5000(i)   = h2gf_est5000.samples_theta{subjindex,i}(15,1);
        end
    elseif config_file == 8 %'estom2mu2';
        for i = 1:1000
            param.prc.mu2.s1000(i) = h2gf_est1000.samples_theta{subjindex,i}(2,1);
            param.prc.om2.s1000(i)  = h2gf_est1000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s1000(i)   = h2gf_est1000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:3000
            param.prc.mu2.s3000(i) = h2gf_est3000.samples_theta{subjindex,i}(2,1);
            param.prc.om2.s3000(i)  = h2gf_est3000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s3000(i)   = h2gf_est3000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:4000
            param.prc.mu2.s4000(i) = h2gf_est4000.samples_theta{subjindex,i}(2,1);
            param.prc.om2.s4000(i)  = h2gf_est4000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s4000(i)   = h2gf_est4000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:5000
            param.prc.mu2.s5000(i) = h2gf_est5000.samples_theta{subjindex,i}(2,1);
            param.prc.om2.s5000(i)  = h2gf_est5000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s5000(i)   = h2gf_est5000.samples_theta{subjindex,i}(15,1);
        end
    elseif config_file == 9 %'estom2mu3';
        for i = 1:1000
            param.prc.mu3.s1000(i) = h2gf_est1000.samples_theta{subjindex,i}(3,1);
            param.prc.om2.s1000(i)  = h2gf_est1000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s1000(i)   = h2gf_est1000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:3000
            param.prc.mu3.s3000(i) = h2gf_est3000.samples_theta{subjindex,i}(3,1);
            param.prc.om2.s3000(i)  = h2gf_est3000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s3000(i)   = h2gf_est3000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:4000
            param.prc.mu3.s4000(i) = h2gf_est4000.samples_theta{subjindex,i}(3,1);
            param.prc.om2.s4000(i)  = h2gf_est4000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s4000(i)   = h2gf_est4000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:5000
            param.prc.mu3.s5000(i) = h2gf_est5000.samples_theta{subjindex,i}(3,1);
            param.prc.om2.s5000(i)  = h2gf_est5000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s5000(i)   = h2gf_est5000.samples_theta{subjindex,i}(15,1);
        end
    elseif config_file == 10 %'estom2om3';
        for i = 1:1000
            param.prc.om2.s1000(i)  = h2gf_est1000.samples_theta{subjindex,i}(13,1);
            param.prc.om3.s1000(i)  = h2gf_est1000.samples_theta{subjindex,i}(14,1);
            param.obs.ze.s1000(i)   = h2gf_est1000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:3000
            param.prc.om2.s3000(i)  = h2gf_est3000.samples_theta{subjindex,i}(13,1);
            param.prc.om3.s3000(i)  = h2gf_est3000.samples_theta{subjindex,i}(14,1);
            param.obs.ze.s3000(i)   = h2gf_est3000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:4000
            param.prc.om2.s4000(i)  = h2gf_est4000.samples_theta{subjindex,i}(13,1);
            param.prc.om3.s4000(i)  = h2gf_est4000.samples_theta{subjindex,i}(14,1);
            param.obs.ze.s4000(i)   = h2gf_est4000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:5000
            param.prc.om2.s5000(i)  = h2gf_est5000.samples_theta{subjindex,i}(13,1);
            param.prc.om3.s5000(i)  = h2gf_est5000.samples_theta{subjindex,i}(14,1);
            param.obs.ze.s5000(i)   = h2gf_est5000.samples_theta{subjindex,i}(15,1);
        end
    elseif config_file == 11 %'estom2sa2';
        for i = 1:1000
            param.prc.sa2.s1000(i) = h2gf_est1000.samples_theta{subjindex,i}(5,1);
            param.prc.om2.s1000(i)  = h2gf_est1000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s1000(i)   = h2gf_est1000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:3000
            param.prc.sa2.s3000(i) = h2gf_est3000.samples_theta{subjindex,i}(5,1);
            param.prc.om2.s3000(i)  = h2gf_est3000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s3000(i)   = h2gf_est3000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:4000
            param.prc.sa2.s4000(i) = h2gf_est4000.samples_theta{subjindex,i}(5,1);
            param.prc.om2.s4000(i)  = h2gf_est4000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s4000(i)   = h2gf_est4000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:5000
            param.prc.sa2.s5000(i) = h2gf_est5000.samples_theta{subjindex,i}(5,1);
            param.prc.om2.s5000(i)  = h2gf_est5000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s5000(i)   = h2gf_est5000.samples_theta{subjindex,i}(15,1);
        end
    elseif config_file == 12 %'estom2sa3';
        for i = 1:1000
            param.prc.sa3.s1000(i) = h2gf_est1000.samples_theta{subjindex,i}(6,1);
            param.prc.om2.s1000(i)  = h2gf_est1000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s1000(i)   = h2gf_est1000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:3000
            param.prc.sa3.s3000(i) = h2gf_est3000.samples_theta{subjindex,i}(6,1);
            param.prc.om2.s3000(i)  = h2gf_est3000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s3000(i)   = h2gf_est3000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:4000
            param.prc.sa3.s4000(i) = h2gf_est4000.samples_theta{subjindex,i}(6,1);
            param.prc.om2.s4000(i)  = h2gf_est4000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s4000(i)   = h2gf_est4000.samples_theta{subjindex,i}(15,1);
        end
        for i = 1:5000
            param.prc.sa3.s5000(i) = h2gf_est5000.samples_theta{subjindex,i}(6,1);
            param.prc.om2.s5000(i)  = h2gf_est5000.samples_theta{subjindex,i}(13,1);
            param.obs.ze.s5000(i)   = h2gf_est5000.samples_theta{subjindex,i}(15,1);
        end
    end
    
    
    
    plot_hist(subjindex, param, parameter_label, parameter_nr, config_file, configtype, eta_label, ...
        h2gf_est1000, h2gf_est3000, h2gf_est4000, h2gf_est5000);
    plot_inf(subjindex, param, parameter_label, parameter_nr, config_file, configtype, eta_label, ...
        h2gf_est1000, h2gf_est3000, h2gf_est4000, h2gf_est5000);
    delete(findall(0,'Type','figure'));
end
end

function plot_hist(subjindex, param, parameter_label, parameter_nr, config_file, eta_label, ...
    configtype, h2gf_est1000, h2gf_est3000, h2gf_est4000, h2gf_est5000)

for j = 1:length(fields(param.prc))
    fig1=figure('rend','painters','pos',[10 10 1300 700],'Name',['Hist srl1; Parameter; ',parameter_label{j}, ' Config: ', configtype ],'NumberTitle','off');
    fig1.Color = [1 1 1];
    
    subplot(2,2,1); fig1=histogram(param.prc.(parameter_label{j}).s1000, 'FaceColor', [0.4 0.5 0.6]);hold on;
    line([h2gf_est1000.summary(subjindex).prc_mean(parameter_nr(j),1), h2gf_est1000.summary(subjindex).prc_mean(parameter_nr(j),1)], ...
        ylim, 'LineWidth', 2, 'Color', 'black');
    title(['nr. of samples: 1000; ', ...
        parameter_label{j}, ' = ', num2str(h2gf_est1000.summary(subjindex).prc_mean(parameter_nr(j),1))]); hold on;
    
    subplot(2,2,2); fig1=histogram(param.prc.(parameter_label{j}).s3000, 'FaceColor', [0.6 0.5 0.4]);hold on;
    line([h2gf_est3000.summary(subjindex).prc_mean(parameter_nr(j),1), h2gf_est3000.summary(subjindex).prc_mean(parameter_nr(j),1)], ...
        ylim, 'LineWidth', 2, 'Color', 'black');
    title(['nr. of samples: 3000; ', ...
        parameter_label{j}, ' = ', num2str(h2gf_est3000.summary(subjindex).prc_mean(parameter_nr(j),1))]); hold on;
    
    subplot(2,2,3); fig1=histogram(param.prc.(parameter_label{j}).s4000, 'FaceColor', [0.4 0.4 0.6]);hold on;
    line([h2gf_est4000.summary(subjindex).prc_mean(parameter_nr(j),1), h2gf_est4000.summary(subjindex).prc_mean(parameter_nr(j),1)], ...
        ylim, 'LineWidth', 2, 'Color', 'black');
    title(['nr. of samples: 4000; ', ...
        parameter_label{j}, ' = ', num2str(h2gf_est4000.summary(subjindex).prc_mean(parameter_nr(j),1))]); hold on;
    
    subplot(2,2,4);fig1=histogram(param.prc.(parameter_label{j}).s5000, 'FaceColor', [0.4 0.5 0.7]);hold on;
    line([h2gf_est5000.summary(subjindex).prc_mean(parameter_nr(j),1), h2gf_est5000.summary(subjindex).prc_mean(parameter_nr(j),1)], ...
        ylim, 'LineWidth', 2, 'Color', 'black');
    title(['nr. of samples: 5000; ', ...
        parameter_label{j}, ' = ', num2str(h2gf_est5000.summary(subjindex).prc_mean(parameter_nr(j),1))]); hold on;
    
    linkaxes;
    suptitle(configtype);
    print(['Hist_srl1_h2gf_',parameter_label{j},'_',configtype,'_eta',eta_label,'_subjnr_',num2str(subjindex)],'-dtiff');
end

fig1=figure('rend','painters','pos',[10 10 1300 700],'Name',['Hist srl1; Parameter; ze, Config: ', configtype ],'NumberTitle','off');
fig1.Color = [1 1 1];
subplot(2,2,1); fig1=histogram(param.obs.ze.s1000, 'FaceColor', [0.4 0.5 0.6]);hold on;
line([h2gf_est1000.summary(subjindex).obs_mean, h2gf_est1000.summary(subjindex).obs_mean], ylim, 'LineWidth', 2, 'Color', 'black');
title(['nr. of samples: 1000; ze = ', num2str(h2gf_est1000.summary(subjindex).obs_mean)]); hold on;

subplot(2,2,2); fig1=histogram(param.obs.ze.s3000, 'FaceColor', [0.6 0.5 0.4]);hold on;
line([h2gf_est3000.summary(subjindex).obs_mean, h2gf_est3000.summary(subjindex).obs_mean], ylim, 'LineWidth', 2, 'Color', 'black');
title(['nr. of samples: 3000; ze = ', num2str(h2gf_est3000.summary(subjindex).obs_mean)]); hold on;

subplot(2,2,3); fig1=histogram(param.obs.ze.s4000, 'FaceColor', [0.4 0.4 0.6]);hold on;
line([h2gf_est4000.summary(subjindex).obs_mean, h2gf_est4000.summary(subjindex).obs_mean], ylim, 'LineWidth', 2, 'Color', 'black');
title(['nr. of samples: 4000; ze = ', num2str(h2gf_est4000.summary(subjindex).obs_mean)]); hold on;

subplot(2,2,4);fig1=histogram(param.obs.ze.s5000, 'FaceColor', [0.4 0.5 0.7]);hold on;
line([h2gf_est5000.summary(subjindex).obs_mean, h2gf_est5000.summary(subjindex).obs_mean], ylim, 'LineWidth', 2, 'Color', 'black');
title(['nr. of samples: 5000; ze = ', num2str(h2gf_est5000.summary(subjindex).obs_mean)]); hold on;
linkaxes;
suptitle(configtype);
print(['Hist_srl1_h2gf_ze_',configtype,'_eta',eta_label,'_subjnr_',num2str(subjindex)],'-dtiff');

end

function plot_inf(subjindex, param, parameter_label, parameter_nr, config_file, configtype, eta_label, ...
    h2gf_est1000, h2gf_est3000, h2gf_est4000, h2gf_est5000)

for j = 1:length(fields(param.prc))
    fig2=figure('rend','painters','pos',[10 10 1300 700],'Name',['Samples srl1; Parameter; ',parameter_label{j}, ' Config: ', configtype ],'NumberTitle','off');
    fig2.Color = [1 1 1];
    subplot(2,2,1); fig2=plot(param.prc.(parameter_label{j}).s1000, 'Color', [0.4 0.5 0.6]); hold on;
    line(xlim, [h2gf_est1000.summary(subjindex).prc_mean(parameter_nr(j),1), h2gf_est1000.summary(subjindex).prc_mean(parameter_nr(j),1)], ...
        'LineWidth', 2, 'Color', 'black');
    title(['nr. of samples: 1000; ', ...
        parameter_label{j}, ' = ', num2str(h2gf_est1000.summary(subjindex).prc_mean(parameter_nr(j),1))]); hold on;
    
    subplot(2,2,2); fig2=plot(param.prc.(parameter_label{j}).s3000, 'Color', [0.6 0.5 0.4]); hold on;
    line(xlim, [h2gf_est3000.summary(subjindex).prc_mean(parameter_nr(j),1), h2gf_est3000.summary(subjindex).prc_mean(parameter_nr(j), ...
        1)], 'LineWidth', 2, 'Color', 'black');
    title(['nr. of samples: 3000; ', ...
        parameter_label{j}, ' = ', num2str(h2gf_est3000.summary(subjindex).prc_mean(parameter_nr(j),1))]); hold on;
    
    subplot(2,2,3); fig2=plot(param.prc.(parameter_label{j}).s4000, 'Color', [0.4 0.4 0.6]); hold on;
    line(xlim, [h2gf_est4000.summary(subjindex).prc_mean(parameter_nr(j),1), h2gf_est4000.summary(subjindex).prc_mean(parameter_nr(j),1)], ...
        'LineWidth', 2, 'Color', 'black');
    title(['nr. of samples: 4000; ', ...
        parameter_label{j}, ' = ', num2str(h2gf_est4000.summary(subjindex).prc_mean(parameter_nr(j),1))]); hold on;
    
    subplot(2,2,4);fig2=plot(param.prc.(parameter_label{j}).s5000, 'Color', [0.4 0.5 0.7]); hold on;
    line(xlim, [h2gf_est5000.summary(subjindex).prc_mean(parameter_nr(j),1), h2gf_est5000.summary(subjindex).prc_mean(parameter_nr(j),1)], ...
        'LineWidth', 2, 'Color', 'black');
    title(['nr. of samples: 5000; ', ...
        parameter_label{j}, ' = ', num2str(h2gf_est5000.summary(subjindex).prc_mean(parameter_nr(j),1))]); hold on;
    linkaxes;
    suptitle(configtype);
    print(['Line_srl1_h2gf_',parameter_label{j},'_',configtype,'_eta',eta_label,'_subjnr_',num2str(subjindex)],'-dtiff');
    %     movefile (['srl_re_h2gf_3l_fixom_eta', eta_label,'_', num2str(NrIter),'.tif'], [maskResFolder, maskModel{1},'/srl_re_h2gf_3l_fixom_eta', eta_label,'_', num2str(NrIter), details.subjname,'.tif']);
end
fig2=figure('rend','painters','pos',[10 10 1300 700],'Name',['Samples srl1; Parameter; ze, Config: ', configtype ],'NumberTitle','off');
fig2.Color = [1 1 1];
subplot(2,2,1); fig2=plot(param.obs.ze.s1000, 'Color', [0.4 0.5 0.6]); hold on;
line(xlim, [h2gf_est1000.summary(subjindex).obs_mean, h2gf_est1000.summary(subjindex).obs_mean], 'LineWidth', 2, 'Color', 'black');
title(['nr. of samples: 1000; ze = ', num2str(h2gf_est1000.summary(subjindex).obs_mean)]); hold on;

subplot(2,2,2); fig2=plot(param.obs.ze.s3000, 'Color', [0.6 0.5 0.4]); hold on;
line(xlim, [h2gf_est3000.summary(subjindex).obs_mean, h2gf_est3000.summary(subjindex).obs_mean], 'LineWidth', 2, 'Color', 'black');
title(['nr. of samples: 3000; ze = ', num2str(h2gf_est3000.summary(subjindex).obs_mean)]); hold on;

subplot(2,2,3); fig2=plot(param.obs.ze.s4000, 'Color', [0.4 0.4 0.6]); hold on;
line(xlim, [h2gf_est4000.summary(subjindex).obs_mean, h2gf_est4000.summary(subjindex).obs_mean], 'LineWidth', 2, 'Color', 'black');
title(['nr. of samples: 4000; ze = ', num2str(h2gf_est4000.summary(subjindex).obs_mean)]); hold on;

subplot(2,2,4);fig2=plot(param.obs.ze.s5000, 'Color', [0.4 0.5 0.7]); hold on;
line(xlim, [h2gf_est5000.summary(subjindex).obs_mean, h2gf_est5000.summary(subjindex).obs_mean], 'LineWidth', 2, 'Color', 'black');
title(['nr. of samples: 5000; ze = ', num2str(h2gf_est5000.summary(subjindex).obs_mean)]); hold on;
linkaxes;
suptitle(configtype);
print(['Line_srl1_h2gf_ze_',configtype,'_eta',eta_label,'_subjnr_',num2str(subjindex)],'-dtiff');
end


