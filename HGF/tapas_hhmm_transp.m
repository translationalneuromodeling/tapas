function [pvec, pstruct] = tapas_hhmm_transp(r, ptrans)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

pvec    = NaN(1,length(ptrans));
pstruct = struct;

% Number of outcomes
m = r.c_prc.n_outcomes;

% Transform back to native space
pvec = tapas_sgm(ptrans,1);

% Fill into tree
cN = r.c_prc.N;
N = {};

pv = pvec;
for id = 1:length(cN)
    N{id}.parent   = cN{id}.parent;
    N{id}.children = cN{id}.children;

    if ~isempty(cN{id}.V)
        N{id}.V = pv(1);
        pv(1) = [];
    else
        N{id}.V = [];
    end

    if ~isempty(cN{id}.A)
        nc = length(cN{id}.children);
        N{id}.A = reshape(pv(1:nc^2),nc,nc);
        pv(1:nc^2) = [];
    else
        N{id}.A = [];
    end

    N{id}.B = cN{id}.B;
end

if ~isempty(pv)
    error('tapas:hgf:hhmm:ParamVecNotMatchingTree', 'Parameter vector does not match node tree.');
end

pstruct.N = N;

return;