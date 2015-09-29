function c = tapas_hmm_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the hidden Markov model (HMM)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The structure returned by fitModel() contains the estimated transition matrix A and the
% estimated prior probabilities ppi of the states
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Config structure
c = struct;

% Model name
c.model = 'hmm';

% Number of hidden states
c.n_states = 2;

% Number of possible outcomes
c.n_outcomes = 2;

% Outcome probabilities, given states
% Columns: states; rows: outcomes
% Columns have to add to 1
%
% In this default example:
% 1st column: "black" urn, 2nd column: "white" urn;
% 1st row: black bead, 2nd row: white bead
c.B = [0.15 0.85; 0.85 0.15];

% Prior probabilities of all states but last.
% ppimuredmu must not add to more than 1.
% The last entry in ppi is determined by ppired because ppi has to sum to 1.
ppiredmu = 0.5;
c.logitppiredmu = logit([ppiredmu], 1);
c.logitppiredsa = [1];

% Reduced transition matrix Ared (all columns but last).
% The rows of Aredmu must not add to more than 1.
% The last column of A is determined by Ared because rows have to sum to 1.
Aredmu = [0.9; 0.1];
Aredsa = [0.1; 0.1];

c.logitAredmu = logit(Aredmu(:)', 1);
c.logitAredsa = Aredsa(:)';

% Gather prior settings in vectors
c.priormus = [
    c.logitppiredmu,...
    c.logitAredmu,...
         ];

c.priorsas = [
    c.logitppiredsa,...
    c.logitAredsa,...
         ];

% Model function handle
c.prc_fun = @hmm;

% Handle to function that transforms perceptual parameters to their native space
% from the space they are estimated in
c.transp_prc_fun = @hmm_transp;

return;
