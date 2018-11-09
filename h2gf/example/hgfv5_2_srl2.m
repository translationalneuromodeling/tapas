% HGF v5.2, using the data: PRSSI, EEG, short version of SRL (srl2)
%
% missed trials are removed and therefore not part of the perceptual model
% =========================================================================
% hgfv5_2_srl2
% =========================================================================

function hgfv5_2_srl2(config_file)

addpath(genpath('/cluster/project/tnu/igsandra/tapas/'));
%% We load the behavioural srl2 data
data_srl2 = tapas_h2gf_load_example_data_srl2();
% Number of subjects
num_subjects = length(data_srl2);

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