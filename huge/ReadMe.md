This toolbox implements variational Bayesian inversion for hierarchical 
unsupervised generative embedding (HUGE). To get started, run the demo script
with the following command:
[DcmInfo, DcmResults] = tapas_huge_demo.m;

The toolbox requires compilation of mex files, which is done automatically.
If you wish to compile manually, use the following command:
mex tapas_huge_int_euler.c
To choose a compiler, use the command:
mex -setup

For more information, read following the paper:
Yao Y, Raman SS, Schiek M, Leff A, Frässle S, Stephan KE (2018). Variational 
Bayesian Inversion for Hierarchical Unsupervised Generative Embedding (HUGE). 
NeuroImage, 179: 604-619
https://doi.org/10.1016/j.neuroimage.2018.06.073

The HUGE toolbox is part of TAPAS, which is released under the terms of the 
GNU General Public Licence (GPL), version 3. For further details, see 
<http://www.gnu.org/licenses/>.

This software is intended for research only. Do not use for clinical purpose. 
Please note that the HUGE toolbox is in an early stage of development. 
Considerable changes are planned for future releases. For support, please 
refer to:
https://github.com/translationalneuromodeling/tapas/issues

Author: Yu Yao (yao@biomed.ee.ethz.ch)
Copyright (C) 2018 Translational Neuromodeling Unit
                   Institute for Biomedical Engineering,
                   University of Zurich and ETH Zurich.