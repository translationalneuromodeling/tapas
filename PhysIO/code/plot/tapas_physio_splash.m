function verbose = tapas_physio_splash(verbose)
%Print splash information
%
%   output = tapas_physio_splash(input)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_splash
%
%   See also
 
% Author:   Lars Kasper
% Created:  2022-02-17
% Copyright (C) 2022 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.
 

if nargin < 1
    verbose.fig_handles = [];
    verbose.level = 0;
end

linkFAQ = 'https://gitlab.ethz.ch/physio/physio-doc/-/wikis/FAQ#3-how-do-i-cite-physio';
linkPaperPhysIO = 'https://doi.org/10.1016/j.jneumeth.2016.10.019';
linkPaperTAPAS = 'https://doi.org/10.3389/fpsyt.2021.680811';
linkPaperHilbertRVT = 'https://doi.org/10.1016/j.neuroimage.2021.117787';
linkGithubTAPASPhysIO = 'https://github.com/translationalneuromodeling/tapas/tree/master/PhysIO';


disp('  _____  _               _____ ____    _______          _ _')
disp(' |  __ \| |             |_   _/ __ \  |__   __|        | | |')              
disp(' | |__) | |__  _   _ ___  | || |  | |    | | ___   ___ | | |__   _____  __')
disp(' |  ___/| ''_ \| | | / __| | || |  | |    | |/ _ \ / _ \| | ''_ \ / _ \ \/ /')
disp(' | |    | | | | |_| \__ \_| || |__| |    | | (_) | (_) | | |_) | (_) >  <') 
disp(' |_|    |_| |_|\__, |___/_____\____/     |_|\___/ \___/|_|_.__/ \___/_/\_\')
disp('                __/ |')                                                     
disp('               |___/')                                                      


fprintf('\n\t This is the Standalone version of the TAPAS PhysIO Toolbox within compiled SPM.');
fprintf('\n\t Please refer to the following FAQ page on how to cite this work:');

% Matlab command window available that renders html
if ~(isdeployed || ismcc) && usejava('Desktop')
    fprintf('\n\n\t <a href="%s">%s</a>', linkFAQ, linkFAQ);
    fprintf('\n\n\t The most relevant references related to this work are:');
    fprintf('\n\t - Main PhysIO paper:               <a href="%s">%s</a>', linkPaperPhysIO, linkPaperPhysIO);
    fprintf('\n\t - Main TAPAS paper:                <a href="%s">%s</a>', linkPaperTAPAS, linkPaperTAPAS);
    fprintf('\n\t - Respiratory Preprocessing/RVT:   <a href="%s">%s</a>', linkPaperHilbertRVT, linkPaperHilbertRVT);
    fprintf('\n\n\t More information and the Matlab version of PhysIO can be found on the TAPAS Github:');
    fprintf('\n\n\t <a href="%s">%s</a>', linkGithubTAPASPhysIO, linkGithubTAPASPhysIO);
    fprintf('\n\n');
else
    fprintf('\n\n\t %s', linkFAQ);
    fprintf('\n\n\t The most relevant references related to this work are:');
    fprintf('\n\t - Main PhysIO paper:               %s', linkPaperPhysIO);
    fprintf('\n\t - Main TAPAS paper:                %s', linkPaperTAPAS);
    fprintf('\n\t - Respiratory Preprocessing/RVT:   %s', linkPaperHilbertRVT);
    fprintf('\n\n\t More information and the Matlab version of PhysIO can be found on the TAPAS Github:');
    fprintf('\n\n\t %s', linkGithubTAPASPhysIO);
    fprintf('\n\n');
end