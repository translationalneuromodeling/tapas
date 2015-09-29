% Simple demonstration of Bayesian mixed-effects inference on the
% accuracy and balanced accuracy. Inferences are based on MCMC
% implementations.
%
% Usage:
%     micp_demo_mcmc

% Kay H. Brodersen, ETH Zurich, Switzerland
% $Id: micp_demo_mcmc.m 16210 2012-05-31 07:04:48Z bkay $
% -------------------------------------------------------------------------
function micp_demo_mcmc
    
    disp(' ');
    disp('MIXED-EFFECTS INFERENCE USING MCMC');
    disp('==================================');
    disp(' ');
    
    % ---------------------------------------------------------------------
    % STEP 1: note down data
    disp('Step 1: noting down classification outcomes');
    ks = [19 41 15 39 39; 41 46 43 48 37];
    ns = [45 51 20 46 58; 55 49 80 54 42];
    disp(['    ks = ',mat2str(ks)]);
    disp(['    ns = ',mat2str(ns)]);
    disp('Press any key to continue...'); pause; disp(' ');
    
    % ---------------------------------------------------------------------
    % STEP 2: visualize data
    disp('Step 2: visualize data');
    micp_demo_mcmc_plot(ks,ns,[],[]);
    disp('Press any key to continue...'); pause; disp(' ');
    
    % ---------------------------------------------------------------------
    % STEP 3: inference
    
    % Initialize
    disp('Step 3: inference');
    try, matlabpool; end
    nSamples = 1000;
    samples_popu = [];
    samples_pijs = [];
    evidences = [];
    names = {}; xlabels = {};
    mo=0;
    
    % Inference on accuracies (beta-binomial)
    mo=mo+1; names{mo} = 'simple beta-binomial model'; xlabels{mo} = 'accuracy';
    disp(['Inference using the ',names{mo},'...']);
    [alphas,betas,pijs] = bicp_sample_ubb_par(sum(ks,1),sum(ns,1),nSamples);
    samples_popu = [samples_popu; alphas./(alphas+betas)];
    if isempty(samples_pijs), samples_pijs(1,:,:)=pijs; else samples_pijs(size(samples_pijs,3)+1,:,:)=pijs; end
    evidences = [evidences; 0];
    
    % Inference on balanced accuracies (twofold beta-binomial)
    mo=mo+1; names{mo} = 'twofold beta-binomial model'; xlabels{mo} = 'balanced accuracy';
    disp(['Inference using the ',names{mo},'...']);
    [alphas_p,betas_p,pijs_p] = bicp_sample_ubb_par(ks(1,:),ns(1,:),nSamples);
    [alphas_n,betas_n,pijs_n] = bicp_sample_ubb_par(ks(2,:),ns(2,:),nSamples);
    samples_popu = [samples_popu; 0.5*(alphas_p./(alphas_p+betas_p) + alphas_n./(alphas_n+betas_n))];
    if isempty(samples_pijs), samples_pijs(1,:,:)=pijs; else samples_pijs(size(samples_pijs,3)+1,:,:)=0.5*(pijs_p+pijs_n); end
    
    % Inference on balanced accuracies (bivariate normal-binomial)
    mo=mo+1; names{mo} = 'combined normal-binomial model'; xlabels{mo} = 'balanced accuracy';
    disp(['Inference using the ',names{mo},'...']);
    [mus,~,pijs] = bicp_sample_bnb_par(ks,ns,nSamples);
    samples_popu = [samples_popu; mean(sigm(mus),1)];
    if isempty(samples_pijs), samples_pijs(1,:,:)=squeeze(mean(pijs,1)); else samples_pijs(size(samples_pijs,3)+1,:,:)=squeeze(mean(pijs,1)); end
    disp('Press any key to continue...'); pause; disp(' ');
    
    % ---------------------------------------------------------------------
    % STEP 4: visualize inferences
    disp('Step 4: visualize inferences');
    micp_demo_mcmc_plot(ks,ns,samples_popu,samples_pijs,'names',names,'xlabels',xlabels,'new_figure',false,'show_data',false,'i',3);
    disp('Press any key to continue...'); pause; disp(' ');
    
    % ---------------------------------------------------------------------
    % STEP 5: Bayesian model comparison (optional)
    disp('Step 5: Bayesian model comparison');
    disp('Computing Bayes factor...');
    logBF = bicp_bms(ks,ns);
    
    disp(' '); disp('Done');
end
