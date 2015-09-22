% Generates a 2 x SUBJECTS matrix of [k+;k-]-samples. For each subject,
% proceeds by drawing a rho (2x1) from N(rho|mu,Sigma), transforming it
% into pi = sigm(rho), and then drawing k (2x1) from Bin(k|pi+) and
% Bin(k|pi-).
%
% Usage:
%     ks = bicp_gen_ks_nb(mu,Sigma,ns)
%     [ks,pijs_true] = bicp_gen_ks_nb(mu,Sigma,ns)
% 
% Arguments:
%     mu,Sigma: population parameters
%     ns: number of trials in each subject
%
% Return values:
%     ks: number of 'correctly classified' trials
%     pijs_true (optional): subject-specific ground-truth accuracies

% Kay H. Brodersen, ETH Zurich, Switzerland
% http://people.inf.ethz.ch/bkay/
% $Id: bicp_gen_ks_bnb.m 13696 2012-01-05 10:36:43Z bkay $
% -------------------------------------------------------------------------
function [ks,pijs_true] = bicp_gen_ks_bnb(mu,Sigma,ns)
    
    % Initialization
    m = size(ns,2);
    ks = NaN(2,m);
    assert(size(mu,1)==2 && size(mu,2)==1);
    assert(ndims(Sigma)==2 && all(size(Sigma)==2));
    pijs_true = NaN(2,m);
    
    % Loop over subjects
    for j = 1:m
        
        % Generate
        rho = mvnrnd(mu,Sigma)';
        pi = sigm(rho);
        ks(1,j) = binornd(ns(1,j),pi(1));
        ks(2,j) = binornd(ns(2,j),pi(2));
        pijs_true(:,j) = pi;
    end
    
end
