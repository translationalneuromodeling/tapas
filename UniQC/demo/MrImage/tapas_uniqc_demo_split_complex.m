% Script demo_split_complex
% splits complex nD data in (n+1)D magn/phase or real/imag
%
%  demo_split_complex
%
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-05-22
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

%
 
 
clear;
close all;
clc; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create complex noise data with imprints
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
nSamples = [48, 48, 9, 4, 3];
data = randn(nSamples);
dataReal = tapas_uniqc_create_image_with_index_imprint(data);
% to change orientation of imprint in imag part
dataImag = permute(tapas_uniqc_create_image_with_index_imprint(data),[2 1 3 4 5]); 
I = MrImage(dataReal+1i*dataImag, ...
    'dimLabels', {'x', 'y', 'z', 't', 'echo'}, ...
    'units', {'mm', 'mm', 'mm', 's', 'ms'}, ...
    'resolutions', [1.5 1.5 3 2 10], 'nSamples', nSamples);

I.real.plot('t',4);
I.imag.plot('t',4);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Split into magn/phase or real/imag
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

I_mp = I.split_complex('mp');
I_mp.plot('t', 4, 'complex_mp', [1 2]);

I_ri = I.split_complex('ri');
I_ri.plot('t', 4, 'complex_ri', [1 2]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Recombine into magn/phase or real/imag
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

I_cpx_mp =  I_mp.combine_complex();
I_cpx_ri =  I_ri.combine_complex();


I_cpx_mp.real.plot('t',4);
I_cpx_mp.imag.plot('t',4);


I_cpx_ri.real.plot('t',4);
I_cpx_ri.imag.plot('t',4);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Try some smoothing...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sI = I.smooth('fwhm',I.dimInfo.resolutions('x'));
sI.real.plot();
sI.imag.plot();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Realign using magnitude
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% default: first echo is representation
[rI, rp] = I.realign();

% second echo 
[r2I, rp2] = I.realign('representationIndexArray', {'echo',2});

% realign each echo individually on itself
[r3I, rp3] = I.realign('representationIndexArray', ...
    {{'echo',1},{'echo',2}, {'echo',3}}, ...
    'applicationIndexArray', ...
    {{'echo',1},{'echo',2}, {'echo',3}});

% check that all three realignments are a bit different
figure('Name', 'Realignment Parameters'); 
plot(rp); hold all;
plot(rp2, '--');
plot(rp3{3}, ':');
