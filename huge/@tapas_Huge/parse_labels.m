function [ tickLabels ] = parse_labels( dcm, labels, idx )
% Generate labels for axis ticks.
% 
% This is a protected method of the tapas_Huge class. It cannot be called
% from outsite the class.
% 

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2020 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <https://www.gnu.org/licenses/>.
% 
% This software is provided "as is", without warranty of any kind, express
% or implied, including, but not limited to the warranties of
% merchantability, fitness for a particular purpose and non-infringement.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is under active development.
% Considerable changes may occur in future releases. For support please
% refer to:
% https://github.com/translationalneuromodeling/tapas/issues
% 

L = size(dcm.c, 2);
R = dcm.n;

% linear connections
a = cell(size(dcm.a));
for rSource = 1:R
    for rTarget = 1:R
        a{rTarget, rSource} = [labels.regions{rSource} ' => ' ...
            labels.regions{rTarget}];
    end
end

% driving connections
c = cell(size(dcm.c));
for lSource = 1:L
    for rTarget = 1:R
        c{rTarget, lSource} = [labels.inputs{lSource} ' => ' ...
            labels.regions{rTarget}];
    end
end

% modulatory connections
b = cell(size(dcm.b));
for l = 1:L
    for rSource = 1:R
        for rTarget = 1:R
            b{rTarget, rSource, l} = [labels.inputs{lSource} ' MOD ' ...
                labels.regions{rSource} ' => ' labels.regions{rTarget}];
        end
    end
end

% nonlinear connections
d = cell(size(dcm.d));
for rMod = 1:size(d,3)
    for rSource = 1:R
        for rTarget = 1:R
            d{rTarget, rSource, rMod} = [labels.regions{rMod} ' MOD ' ...
                labels.regions{rSource} ' => ' labels.regions{rTarget}];
        end
    end
end

% hemodynamic parameters
hemo = cell(R, 2);
for r = 1:R
    hemo{r, 1} = ['transit ' labels.regions{r}];
    hemo{r, 2} = ['decay ' labels.regions{r}];
end

tmp = [a(:); b(:); c(:); d(:); hemo(:); {'epsilon'}];
tickLabels = tmp([idx.clustering(:); idx.homogenous(:)]);

end

