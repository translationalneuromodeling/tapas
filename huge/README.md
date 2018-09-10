Hierarchical Unsupervised Generative Embedding Toolbox (HUGE)
=============================================================

This toolbox implements variational Bayesian inversion for hierarchical 
unsupervised generative embedding (HUGE). To get started, read the 
[tutorial](huge_tutorial.html), which you can also open in Matlab:
```
    open huge_tutorial.html
```
or run the demo script:
```
    open tapas_huge_demo.m
```

The supported interface is:

```
[DcmResults] = tapas_huge_invert(DCM, K, priors, verbose, randomize, seed)
INPUT:
  DCM       - cell array of DCM in SPM format
  K         - number of clusters (set K to one for empirical Bayes)

OPTIONAL INPUT:
  priors    - model priors stored in a struct containing the
              following fields:
      alpha:         parameter of Dirichlet prior (alpha_0 in Fig.1 of
                     REF [1])
      clustersMean:  prior mean of clusters (m_0 in Fig.1 of REF [1])
      clustersTau:   tau_0 in Fig.1 of REF [1]
      clustersDeg:   degrees of freedom of inverse-Wishart prior (nu_0 in
                     Fig.1 of REF [1]) 
      clustersSigma: scale matrix of inverse-Wishart prior (S_0 in Fig.1
                     of REF [1]) 
      hemMean:       prior mean of hemodynamic parameters (mu_h in Fig.1
                     of REF [1]) 
      hemSigma:      prior covariance of hemodynamic parameters(Sigma_h
                     in Fig.1 of REF [1]) 
      noiseInvScale: prior inverse scale of observation noise (b_0 in
                     Fig.1 of REF [1]) 
      noiseShape:    prior shape parameter of observation noise (a_0 in
                     Fig.1 of REF [1])
              (you may use tapas_huge_build_prior(DCM) to generate this
              struct)
  verbose   - activates command line output (prints free energy
              difference, default: false)
  randomize - randomize starting values (default: false). WARNING:
              randomizing starting values can cause divergence of DCM.
  seed      - seed for random number generator

OUTPUT:
  DcmResults - struct used for storing the results from VB. Posterior
               parameters are stored in DcmResults.posterior, which is a
               struct containing the following fields:
      alpha:               parameter of posterior over cluster weights
                           (alpha_k in Eq.(15) of REF [1]) 
      softAssign:          posterior assignment probability of subjects 
                           to clusters (q_nk in Eq.(18) in REF [1])
      clustersMean:        posterior mean of clusters (m_k in Eq.(16) of
                           REF [1]) 
      clustersTau:         tau_k in Eq.(16) of REF [1]
      clustersDeg:         posterior degrees of freedom (nu_k in Eq.(16) 
                           of REF [1])
      clustersSigma:       posterior scale matrix (S_k in Eq.(16) of
                           REF [1]) 
      logDetClustersSigma: log-determinant of S_k
      dcmMean:             posterior mean of DCM parameters (mu_n in
                           Eq.(19) of REF [1])  
      dcmSigma:            posterior covariance of hemodynamic
                           parameters (Sigma_n in Eq.(19) of REF [1]) 
      logDetPostDcmSigma:  log-determinant of Sigma_n
      noiseInvScale:       posterior inverse scale of observation noise
                           (b_n,r in Eq.(21) of REF [1]) 
      noiseShape:          posterior shape parameter of observation noise
                           (a_n,r in Eq.(21) of REF [1])
      meanNoisePrecision:  posterior mean of precision of observation
                           noise (lambda_n,r in Eq.(23) of REF [1])
      modifiedSumSqrErr:   b'_n,r in Eq.(22) of REF [1]
```

The toolbox requires compilation of mex files, which is done automatically.
If you wish to compile manually, set the current directory of your Matlab 
session to the folder containing the HUGE toolbox and use the following 
command:
```
    mex tapas_huge_int_euler.c
```
To choose a compiler, use the command:
```
    mex -setup
```

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

Author
------
Yu Yao (yao@biomed.ee.ethz.ch)  

Copyright (C) 2018 Translational Neuromodeling Unit  
                   Institute for Biomedical Engineering,  
                   University of Zurich and ETH Zurich.