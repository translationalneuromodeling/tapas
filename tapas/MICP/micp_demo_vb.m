% Simple demonstration of Bayesian mixed-effects inference on the
% accuracy and balanced accuracy. Inferences are based on variational Bayes
% (VB) implementations.
%
% Usage:
%     micp_demo_vb

% Kay H. Brodersen, ETH Zurich, Switzerland
% $Id: micp_demo_vb.m 16210 2012-05-31 07:04:48Z bkay $
% -------------------------------------------------------------------------
function micp_demo_vb
    
    disp(' ');
    disp('MIXED-EFFECTS INFERENCE USING VB');
    disp('================================');
    disp(' ');
    
    % ---------------------------------------------------------------------
    % STEP 1: note down data
    disp('Step 1: noting down classification outcomes');
    data.ks = [19 41 15 39 39; 41 46 43 48 37];
    data.ns = [45 51 20 46 58; 55 49 80 54 42];
    disp(['    ks = ',mat2str(data.ks)]);
    disp(['    ns = ',mat2str(data.ns)]);
    disp('Press any key to continue...'); pause; disp(' ');
    
    % ---------------------------------------------------------------------
    % STEP 2: visualize data
    disp('Step 2: visualize data');
    micp_demo_vb_plot(data,[]);
    disp('Press any key to continue...'); pause; disp(' ');
    
    % ---------------------------------------------------------------------
    % STEP 3: inference
    disp('Step 3: inference');
    infc.name = 'variational twofold normal-binomial';
    infc.xlabel = 'balanced accuracy';
    infc.qp = vbicp_unb(data.ks(1,:),data.ns(1,:));
    infc.qn = vbicp_unb(data.ks(2,:),data.ns(2,:));
    disp('Posterior mean of the population mean balanced accuracy:');
    disp(num2str(logitnavgmean(infc.qp.mu_mu,sqrt(1/infc.qp.eta_mu),infc.qn.mu_mu,sqrt(1/infc.qn.eta_mu))));
    disp('Press any key to continue...'); pause; disp(' ');
    
    % ---------------------------------------------------------------------
    % STEP 4: visualize inferences
    disp('Step 4: visualize inferences');
    micp_demo_vb_plot(data,infc,'new_figure',false,'show_data',false,'i',4);
    disp(' '); disp('Done');
end
