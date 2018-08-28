CONTRIBUTING Guide for Public TAPAS Contributors
================================================

How to Contribute
-----------------

- Start a discussion on feature requests or report bugs via [GitHub Issues](https://github.com/translationalneuromodeling/tapas/issues)
    - Please **do not** submit code via the Issue system (e.g., as file attachments or posted snippets), but use *pull requests* (s.b.)
- Please provide code for TAPAS exclusively via submitting *Pull Requests* on GitHub
    - Please accept the [Contributor License Agreement](Contributor-License-Agreement) by 
        1. adding your name (and affiliation, plus e-mail/GitHub username that serve to identify you unambiguously) to the `Contributor-License-Agreement.md` file in the `tapas/` main folder and 
        2. submitting this updated file with your pull request.
    - This procedure ensures the continued legal maintainability of the TAPAS software suite.
    - Have a look at [Wikipedia](https://en.wikipedia.org/wiki/Contributor_License_Agreement) for details on the rationale and legal implications of a CLA. Here is a quote of the essence:
    
    > The purpose of a CLA is to ensure that the guardian of a project's outputs has
    > the necessary ownership or grants of rights over all contributions to allow 
    > them to distribute under the chosen license. 


Coding and Style Guidelines
---------------------------

- In general, try to adhere to existing style guides that are used by many programmers
    - e.g., the [Matlab Style Guidelines V2.0](https://ch.mathworks.com/matlabcentral/fileexchange/46056-matlab-style-guidelines-2-0) by Richard Johnson
- Readability: Keep in mind that software should be used, and maybe extended by others!
- Have a look at the specific toolbox within TAPAS you want to contribute to and
    - try to adhere to the coding style of that toolbox
    - document well, and use consistent verbose headers for functions
    - check whether the toolbox has its own `CONTRIBUTING.md` document for further specific requests.


