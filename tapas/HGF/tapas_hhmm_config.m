function c = tapas_hhmm_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the hierarchical hidden Markov model (HHMM)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This implementation follows the exposition in Fine, Singer, & Tishby (1998). The hierarchical
% hidden Markov model: Analysis and Applications. Machine Learning, 32, 41–62.
%
% The structure returned by fitModel() contains the estimated transition matrices A (responsible for
% horizontal transitions), the vertical transition probabilities V and the outcome contingencies B.
%
% The configuration is best explained by way of example, so please see the template below. Briefly,
% the model tree is configured as a cell array N of nodes. Each node therefore has its unique id,
% namely its position in that array. Id 1 (the first position) is reserved for the root node. Apart
% from that, the id has nothing to do with the structure of the tree. That is defined in the
% individual nodes by setting N{id}.parent and N{id}.children. While setting both is strictly
% speaking redundant, it is required because it is convenient and provides an additional basis
% for sanity checking. That said, while some sanity checking is done, it is your responsibility
% to define a tree that makes sense. The nodes at the bottom of the tree (the ones generating
% output) are called production nodes.
%
% Estimated quantities are the vertical transition probabilities V (called pi in Fine et
% al. (1998)), the transition matrices A, and the outcome contingencies B of the production
% nodes. Each node, except the root, has N{id}.V, containing the prior for V (Gaussian in
% logit space) defined by the mean logitmu and the variance (not standard deviation)
% logitsa. Likewise, each node, except production nodes, has N{id}.A, containing the prior of the
% transition matrix between its children, also in logit space. Production nodes have N{id}.B, the
% outcome contingency vector. V is an empty array for the root node, A is empty for production
% nodes, and B is empty for non-production nodes.
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Config structure
c = struct;

% Model name
c.model = 'hhmm';

% Number of possible outcomes
c.n_outcomes = 2;

% Model tree (N is for node)
%
% The root node is the first element of the cell array of nodes
c.N{1} = struct;

% The root node has no parent
c.N{1}.parent = [];

% These nodes are the root's children
c.N{1}.children = [2, 3];

% The vertical transition probability into the root node is empty
c.N{1}.V = [];

% Prior of transition matrix A between the children of the root node.
% Its elements have Gaussian priors in logit space. Inf in logit space corresponds to 1 in native
% space; likewise, -Inf corresponds to 0.
c.N{1}.A.logitmu = [-Inf, Inf; Inf, -Inf];
c.N{1}.A.logitsa = [   0,   0;   0,    0];

% The outcome contingencies B of the root node are empty because the root is not a production
% node
c.N{1}.B = [];

% The further nodes follow the same pattern (see explanation at the top of the file)
% Adapt as needed below. Description of the nodes is just an example.

% The low volatility regime
c.N{2}.parent = 1;
c.N{2}.children = [4, 5];
c.N{2}.V.logitmu = logit(0.5,1);
c.N{2}.V.logitsa = 0;
c.N{2}.A.logitmu = [logit(0.9,1), logit(0.05,1); logit(0.05,1), logit(0.9,1)];
c.N{2}.A.logitsa = [           0,           0.1;           0.1,            0];
c.N{2}.B = [];

% The high volatility regime
c.N{3}.parent = 1;
c.N{3}.children = [6, 7];
c.N{3}.V.logitmu = logit(0.5,1);
c.N{3}.V.logitsa = 0;
c.N{3}.A.logitmu = [logit(0.6,1), logit(0.35,1); logit(0.35,1), logit(0.6,1)];
c.N{3}.A.logitsa = [           0,           0.1;           0.1,            0];
c.N{3}.B = [];

% Black urn (low volatility)
c.N{4}.parent = 2;
c.N{4}.children = [];
c.N{4}.V.logitmu = logit(0.5,1);
c.N{4}.V.logitsa = 0;
c.N{4}.A = [];
c.N{4}.B = [0.15, 0.85];

% White urn (low volatility)
c.N{5}.parent = 2;
c.N{5}.children = [];
c.N{5}.V.logitmu = logit(0.5,1);
c.N{5}.V.logitsa = 0;
c.N{5}.A = [];
c.N{5}.B = [0.85, 0.15];

% Black urn (high volatility)
c.N{6}.parent = 3;
c.N{6}.children = [];
c.N{6}.V.logitmu = logit(0.5,1);
c.N{6}.V.logitsa = 0;
c.N{6}.A = [];
c.N{6}.B = [0.15, 0.85];

% White urn (high volatility)
c.N{7}.parent = 3;
c.N{7}.children = [];
c.N{7}.V.logitmu = logit(0.5,1);
c.N{7}.V.logitsa = 0;
c.N{7}.A = [];
c.N{7}.B = [0.85, 0.15];

% END OF CONFIGURATION
% ~~~~~~~~~~~~~~~~~~~~

% Vectorize priors by walking through tree
mus = [];
sas = [];
for id = 1:length(c.N)
    if ~isempty(c.N{id}.V)
        mus = [mus, c.N{id}.V.logitmu];
        sas = [sas, c.N{id}.V.logitsa];
    end

    if ~isempty(c.N{id}.A)
        mus = [mus, c.N{id}.A.logitmu(:)'];
        sas = [sas, c.N{id}.A.logitsa(:)'];
    end
end


% Gather prior settings in vectors
c.priormus = mus;
c.priorsas = sas;

% Model function handle
c.prc_fun = @hhmm;

% Handle to function that transforms perceptual parameters to their native space
% from the space they are estimated in
c.transp_prc_fun = @hhmm_transp;

return;
