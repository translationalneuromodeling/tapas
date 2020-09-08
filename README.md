![TAPAS Logo](misc/TapasLogo.png?raw=true "TAPAS Logo")

*Version 4.4.0*

T  A  P  A  S - Translational Algorithms for Psychiatry-Advancing Science.
========================================================================

This file describes the installation and usage of the software.
Full details can be found on the TNU website:
                 http://www.translationalneuromodeling.org/tapas/


-----------
DESCRIPTION
-----------

TAPAS is a collection of algorithms and software tools developed by the
Translational Neuromodeling Unit (TNU, Zurich) and collaborators. The goal of
these tools is to support clinical neuromodeling, particularly computational
psychiatry, computational neurology, and computational psychosomatics.

Currently, TAPAS includes the following packages:

- [ceode](ceode/README.md): Continuous Extension of ODE methods. A toolbox for robust estimation of convolution based Dynamic Causal Models (DCMs) for evoked responses (ERPs). 
- [HGF](HGF/README.md): The Hierarchical Gaussian Filter; Bayesian inference
  on computational processes from observed behaviour.
- [HUGE](huge/README.md): Variational Bayesian inversion for hierarchical
unsupervised generative embedding (HUGE).
- [MICP](MICP/Readme%20for%20MATLAB.pdf): Bayesian Mixed-effects Inference for Classification Studies.
- [MPDCM](mpdcm/README.md): Massively Parallel DCM; Efficient integration of DCMs using massive parallelization.
- [PhysIO](PhysIO/README.md): Physiological Noise Correction for fMRI.
- [rDCM](rDCM/README.md): DCM based, efficient inference on effective brain connectivity for fMRI.
- [SEM](sem/README.md): SERIA Model for Eye Movements (saccades and anti-saccades) and Reaction Times.
- [VBLM](VBLM/README.txt): Variational Bayesian Linear Regression.

And the following tasks:
- [FDT](task/FDT/README.md): Filter Detection Task.

TAPAS is written in MATLAB and distributed as open source code under
the GNU General Public License (GPL, Version 3).


------------
INSTALLATION
------------

TAPAS is a collection of toolboxes written in MATLAB (Version R2016b). The key
requirement is the installation of MATLAB (produced by The MathWorks, Inc.
Natick, MA, USA. http://www.mathworks.com/).

Please download TAPAS from our
[Github Release Page](https://github.com/translationalneuromodeling/tapas/releases).

To add the TAPAS directory to the MATLAB path, run the script `tapas_init.m` in
the directory where tapas is installed/extracted.

For the individual toolboxes included in TAPAS, please refer to their
documentation (s.b.) for specific installation instructions.

To download the example data please use `tapas_download_example_data()` from
the matlab console.

-------------
DOCUMENTATION
-------------

- The latest documentation of TAPAS can be found in this README and on the
  [GitHub Wiki](https://github.com/translationalneuromodeling/tapas/wiki) of the
  [TAPAS GitHub page](https://github.com/translationalneuromodeling/tapas)
- In general, each toolbox comes with their own documentation as Wiki, PDF,
  matlab tutorials etc.
- Documentation for older versions (<= 2.7.0.0) is provided on the
  [TNU page](https://www.tnu.ethz.ch/de/software/tapas/documentations)


-------
SUPPORT
-------

- Please submit bug reports, feature requests, code improvements etc. via the
  [Issues](https://github.com/translationalneuromodeling/tapas/issues) Forum pages
  on GitHub (you will need a GitHub account.
- This issue forum is searchable, so please have a look if your question has
  been asked before.
- For older versions or more general questions, please also have a look at our
  now deprecated E-Mail List, which has a searchable [Archive](https://sympa.ethz.ch/sympa/arc/tapas)


-------
Cite Me
-------

Information about citations and current version can be printed from matlab with
the command: `tapas_version(1);`

---------------
Current release
---------------

Information about changes in the current release can be found in the [CHANGELOG.md](CHANGELOG.md)
file.

All recent stables releases can be downloaded from our
[TAPAS Github Release Page](https://github.com/translationalneuromodeling/tapas/releases).


-------
LICENSE
-------

This software is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this software. If not, see http://www.gnu.org/licenses/.


------------------
CONDITIONS OF USE
------------------

The Software is distributed "AS IS" under the GNU General Public License (GPL, Version 3) license solely for non-commercial use.  On accepting these conditions, the licensee understand that no condition is made or to be implied, nor is any warranty given or to be implied, as to the accuracy of the Software, or that it will be suitable for any particular purpose or for use under any specific conditions. Furthermore, the software authors and the University of Zurich disclaim all responsibility for the use which is made of the Software. It further disclaims any liability for the outcomes arising from using the Software.

The Licensee agrees to indemnify the software authors and the TNU and hold them harmless from and against any and all claims, damages and liabilities asserted by third parties (including claims for negligence) which arise directly or indirectly from the use of the Software or the sale of any products based on the Software.

No part of the Software may be reproduced, modified, transmitted or transferred in any form or by any means, electronic or mechanical, without the express permission of the University of Zurich. The permission of the University is not required if the said reproduction, modification, transmission or transference is done without financial return, the conditions of this License are imposed upon the receiver of the product, and all original and amended source code is included in any transmitted product. You may be held legally responsible for any copyright infringement that is caused or encouraged by your failure to abide by these terms and conditions.

You are not permitted under this License to use this Software commercially. Use for which any financial return is received shall be defined as commercial use, and includes (1) integration of all or part of the source code or the Software into a product for sale or license by or on behalf of Licensee to third parties or (2) use of the Software or any derivative of it for research with the final aim of developing software products for sale or license to a third party or (3) use of the Software or any derivative of it for research with the final aim of developing non-software products for sale or license to a third party, or (4) use of the Software to provide any service to an external organisation for which payment is received. If you are interested in using the Software commercially, please contact Unitectra (http://www.unitectra.ch/en).

