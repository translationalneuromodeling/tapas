# BREATHING LEARNING TASK


VERSION: 0.1.0

Author: Olivia Harrison and Frederike Petzschner

Created: 18/02/2021

This software is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

See the GNU General Public License for more details: <http://www.gnu.org/licenses/>

-----

### REREFENCING

If you use this task, please reference Harrison et al., "Interoception of breathing and its relationship with anxiety" (in preparation).

If you use the inspiratory resistance circuit designed for this task, please reference Rieger et al. (2020): https://doi.org/10.3389/fnhum.2020.00161

-----

### GENERAL OVERVIEW

This task is a breathing learning task, which operates under a simple associative learning paradigm. One of two (paired) shapes are presented at any trial, with one of these shapes indicating an 80% chance of a inspiratory resistance on that trial, and the other shape indicating a 20% chance of inspiratory resistance. Participants are explicitly informed that the shapes act as a pair, and their associations with resistance can only ever swap (i.e. one shape will always indicate 80% and the other 20%, but they can change places). The output of this task can then be used to assess learning rate etc., via fitting models such as the Rescorla Wagner model.

-----

### IMPORTANT NOTES

  1) This task requires MATLAB and PsychToolBox to be installed to run. MATLAB2017b and PsychToolBox-3 have been tested. If you would like to set up the task to wait for an external trigger (e.g. from an MRI scanner) before starting, you will also need Cogent installed.
  2) Before running this task, you will need to download additional resources. To do this, start MATLAB, set your current working directory to the `BLT` folder and execute the command `tapas_BLT_setup()`. Note: this commands will require a working internet connection and may take some time to finish.
  3) You need to run the task from the main BLT folder in MATLAB. If at any point you should need to terminate the task, use the escape button or 'control' + 'c' followed by 'sca' to exit the loop. All data up to that point will be saved in the specified output file.
  4) To run the task, type the name of the main function at the terminal (tapas_BLT). You will be prompted to enter the experiment mode you would like to use, where the options are:

     * debug: This uses a small screen with limited trials for testing the task.
     * train_4: This is a sequence of 4 trials that can be used for training purposes (full screen).
     * train_6: This is a sequence of 6 trials that can be used for training purposes (full screen).
     * task: This is the full task run of 80 trials (full screen).
  5) You will also be prompted to enter the cue type, where there are four options to counterbalance the cue colour and left/right placement of Yes/No answers in case randomisation is required. The cue type options are:

     * 1: This uses the green cue (in the task and debug modes) or red cue (in the training modes) as first paired with 80% probability of resistance, with the answer 'Yes' on the left.
     * 2: This uses the green cue (in the task and debug modes) or red cue (in the training modes) as first paired with 80% probability of resistance, with the answer 'Yes' on the right.
     * 3: This uses the yellow cue (in the task and debug modes) or blue cue (in the training modes) as first paired with 80% probability of resistance, with the answer 'Yes' on the left.
     * 4: This uses the yellow cue (in the task and debug modes) or blue cue (in the training modes) as first paired with 80% probability of resistance, with the answer 'Yes' on the right.
  6) The main setup parameters are saved in the BLT_initParams.m script. You may choose to alter some of the following task options:

     * Binary vs. sliding scale predictions: The task can be set up such that participants can make their predictions using Yes/No binary answers or a sliding scale of how likely they think a resistance will be following the cue.
     * Binary vs. sliding scale answers: The task can be set up such that participants can be asked to report if there was a resistance (following the stimulus presentation) using a binary Yes/No answer, or they can be asked to rate how difficult they found the resistance on a sliding scale.
     * You may wish to hide the cursor from view if you are running the task using only one screen.
     * You will most likely need to adjust any keyboard and/or button box settings to suit your specific setup.
     * If you would like use Cogent to wait for a scanner trigger to start the task (via a signal from a parallel port), you can specify params.doMRI to = 1.
     * You may also need to change the text size (in BLT_initParams.m) and the anchor text position (in BLT_runRating.m) to optimise text size and position for any sliding scale questions according to your screen size.
     * Any task durations etc. can also be changed in the BLT_initParams.m script.
  7) The administration of inspiratory resistances can be automated using the circuit described in Rieger et al. (2020): https://doi.org/10.3389/fnhum.2020.00161. If you use the resistance circuit setup, you will need to ensure there is a parallel port available to control the circuit valves. You will then need to:

     * Download the required self-installing system driver (named inpoutx64.dll) and save it to your computer (instructions and download can be found here: http://apps.usd.edu/coglab/psyc770/IO64.html).
         * Note 1: Self-installation of the driver requires that MATLAB is first run with administrator privileges. Once installed, MATLAB can be run as a non-administrator.
         * Note 2: Because the inpoutx64.dll was compiled using Visual Studio, the Microsoft Visual C++ 2005 SP1 Redistributable (x64) Package must be installed on your computer. Use the Control Panel to see if it is already installed.  If not, the installer application can be downloaded from Microsoft at http://www.microsoft.com/download/en/details.aspx?displaylang=en&id=18471.
     * Once the driver is installed you will need to find the parallel port address for your computer, and then specify this address in the BLT_initparams.m script (params.port_address).

-----