> ## 🚨 **Important** 🚨 <br> 
> ## TAPAS has moved to a new [location](https://github.com/ComputationalPsychiatry). This repository is no longer maintained.

![TAPAS Logo](misc/TapasLogo.png?raw=true "TAPAS Logo")

*Version 6.1.0*

T  A  P  A  S - Translational Algorithms for Psychiatry-Advancing Science.
========================================================================

This file describes the installation and usage of the software.
Full details can be found on the TNU website:
                 http://www.translationalneuromodeling.org/tapas/


-----------
DESCRIPTION
-----------

TAPAS is a collection of algorithms and software tools developed by the
Translational Neuromodeling Unit (TNU, Zurich) and collaborators. These 
tools are intended to support the development of computational assays 
(Translational Neuromodeling) and their clinical application in 
Computational Psychiatry, Neurology and Psychosomatics.

Currently, TAPAS includes the following packages:

- [ceode](ceode/README.md): Continuous Extension of ODE methods. A toolbox for robust estimation of convolution based Dynamic Causal Models (DCMs) for evoked responses (ERPs). 
- [genbed](genbed/README.md): A Python package for data exploration and classification as part of the **gen**erative em**bed**ding pipeline.
- [HGF](HGF/README.md): The Hierarchical Gaussian Filter; Bayesian inference
  on computational processes from observed behaviour.
- [HUGE](huge/README.md): Variational Bayesian inversion for hierarchical
unsupervised generative embedding (HUGE).
- [PhysIO](PhysIO/README.md): Physiological Noise Correction for fMRI.
- [rDCM](rDCM/README.md): Regression dynamic causal modeling; efficient inference on whole-brain effective connectivity from fMRI data.
- [SEM](sem/README.md): SERIA Model for Eye Movements (saccades and anti-saccades) and Reaction Times.

And the following tasks:
- [FDT](task/FDT/README.md): Filter Detection Task.
- [BLT](task/BLT/README.md): Breathing Learning Task.


TAPAS also includes beta versions of the following toolboxes. Please note that these toolboxes have not been extensively tested and are still in active development:
- [UniQC](UniQC/README.md): unified neuroimaging quality control.


TAPAS is written in MATLAB and Python and distributed as open source code under
the GNU General Public License (GPL, Version 3).

If you cannot find the toolbox you were looking for in the downloaded version, 
you might find it in the [TAPAS Legacy repository](https://tnurepository.ethz.ch/TNU/tapas/-/tree/development).

------------
INSTALLATION
------------

TAPAS is a collection of toolboxes written in MATLAB (Version R2016b) and Python. 
- The key requirement is the installation of MATLAB (produced by The MathWorks, Inc.
Natick, MA, USA. http://www.mathworks.com/). 
- Toolboxes written in Python currently include: 
    - genbed: For requirements see the genbed [documentation](genbed/README.md).

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
  [TAPAS GitHub page](https://github.com/translationalneuromodeling/tapas).
- In general, each toolbox comes with their own documentation as Wiki, PDF,
  matlab tutorials etc.
- Documentation for older versions (<= 2.7.0.0) is provided on the
  [TNU page](https://www.tnu.ethz.ch/de/software/tapas/documentations).


-------
SUPPORT
-------

- Please submit bug reports, feature requests, code improvements etc. via the
  [Issues](https://github.com/translationalneuromodeling/tapas/issues) Forum pages
  on GitHub (you will need a GitHub account.
- This issue forum is searchable, so please have a look if your question has
  been asked before.
- For older versions or more general questions, please also have a look at our
  now deprecated E-Mail List, which has a searchable [Archive](https://sympa.ethz.ch/sympa/arc/tapas).


-------
CITE ME
-------

When using a toolbox which is part of the TAPAS collection, please cite the paper(s) mentioned in the README of the respective toolbox and the paper below describing the entire collection.

You can include for example the following snippet in your Methods section:

> The analysis was performed using the Matlab [NAME-OF-THE-TOOLBOX] Toolbox ([CITATION(S)-OF-TOOBOX], open-source code available as part of the TAPAS software collection: [1] / https://www.translationalneuromodeling.org/tapas)

[1] Frässle, S., Aponte, E.A., Bollmann, S., Brodersen, K.H., Do, C.T., Harrison, O.K., Harrison, S.J., Heinzle, J., Iglesias, S., Kasper, L., Lomakina, E.I., Mathys, C., Müller-Schrader, M., Pereira, I., Petzschner, F.H., Raman, S., Schöbi, D., Toussaint, B., Weber, L.A., Yao, Y., Stephan, K.E.: TAPAS: An Open-Source Software Package for Translational Neuromodeling and Computational Psychiatry, Frontiers in Psychiatry 12, 857, 2021. https://doi.org/10.3389/fpsyt.2021.680811

For additional information concerning citations and the current TAPAS version, enter `tapas_version(1);` in your MATLAB command line.

---------------
CURRENT RELEASE
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

