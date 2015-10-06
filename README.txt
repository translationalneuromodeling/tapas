Version 2.2.0.0

________________________________________________________________________

     T  A  P  A  S - TNU Algorithms for Psychiatry-Advancing Science.
________________________________________________________________________


This file describes the installation and usage of the software.
Full details can be found on the TNU website:
                 http://www.translationalneuromodeling.org/tapas/

-----------
DESCRIPTION
-----------

TAPAS is a collection of algorithms and software tools that are 
developed by the Translational Neuromodeling Unit (TNU) at Zurich. 
These tools have been developed to support translational neuroscience, 
particularly concerning the application of neuroimaging and 
computational modeling to research questions in psychiatry and 
neurology. Problems that can be addressed by tools in TAPAS presently include:

    - Correction of physiological noise in fMRI data.
    - Bayesian inference on computational processes from observed behaviour.
    - Bayesian mixed-effects inference for classification studies.
    - Variational-Bayes Linear Regression.
    - Efficient integration of the DCMs using massive parallelization.

TAPAS is written in MATLAB and distributed as open source code under 
the GNU General Public License (GPL, Version 3).

------------
INSTRUCTIONS                                                            
------------

TAPAS is a collection of toolboxes written in MATLAB (Version R2012b). The key 
requirement is the installation of MATLAB (produced by The MathWorks, Inc.  
Natick, MA, USA. http://www.mathworks.com/).

To add the TAPAS directory to the MATLAB path, run the script tapas_init.m in the directory where tapas is installed/extracted. 

-------------
DOCUMENTATION
-------------

The TNU website contains all the necessary documentation, user manuals,
publications relevant to the software. 
           http://www.translationalneuromodeling.org/tapas-documentation/

For queries and discussions please join the mailing list at http://sympa.ethz.ch/sympa/info/tapas 
Bugs in the software can be emailed directly to: tapas-bugs@biomed.ee.ethz.ch


------------------------
Changes in this version
------------------------

Major changes in the mpdmc toolbox

Support for non linear DCM
Model parameters are stored as sparse matrices
Fully tested
This version corresponds to the paper doi:10.1016/j.jneumeth.2015.09.009

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

