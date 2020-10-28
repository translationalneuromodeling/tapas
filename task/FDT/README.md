# FILTER DETECTION TASK


VERSION: 0.2.2

Author: Olivia Harrison

Created: 14/08/2018

This software is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

See the GNU General Public License for more details: <http://www.gnu.org/licenses/>

-----

Updated to 0.2.2: 27/07/2020

- Paired session difference analysis option added to the toolbox

Updated to 0.2.1: 22/06/2020

- Added error to avoid over-writing files with the same PPID specified

- Information regarding download or location of freely-available high quality pink noise files added

Updated to 0.2.0: 02/09/2019

- Regression analysis option added to the toolbox

- More highly regularised roving staircase with a tighter accuracy band (70-75% range) and less tolerant error risk (0.3 from 0.2)

- Criterion representation fixed (c < 0 indicates bias towards yes)

Updated to 0.1.3: 05/07/2019

- Added option to run with a fixed or roving staircase

Updated to 0.1.2: 17/05/2019

- Bug fixed in `filter_detection_task_fix` script (typo) that caught script

Updated to 0.1.1: 25/11/2018

- 2IFC Task option included as an alternative

- Option to specify upper and lower confidence bounds

- Automatic avoidance of zero filters during main task

-----


### GENERAL OVERVIEW

This task is a breathing detection task that aims to understand how sensitive an individual is to very small resistances to their inspiration, and any biases they may have towards over- or under- reporting these changes (if run as a yes/no task - default setting). Alternatively, this task can be run as a two-interval forced choice task, to create more compatibility if needing to compare the results with other forced-choice tasks.

Furthermore, this task aims to quantify how 'metacognitively aware' and efficient a participant is, or how much their confidence in their decisions relates to the accuracy of their performance. For example, an individual with better metacognitive scores will report higher confidence when they are correct in their perceptual decision, and lower confidence when they are incorrect.

**THIS TASK AIMS TO DETERMINE:**
  1) The number of breathing filters a participant is able to discriminate from a dummy filter with ~60-85% accuracy.
  2) The participant's discrimination metric (d') and bias criterion (c) for reporting breathing resistance, using signal detection theory.
  3) The participant's metacognitive awareness and efficiency about their performance, in the form of both average confidence over trials and the metacognitive efficiency metric of meta-d'/d'.

-----

### EXPERIMENTER INSTRUCTIONS

**TO RUN THIS TASK:**
  1) Copy the entire set of files to a location on your computer and add all folders and subfolders to your matlab path. Open the `filter_detection_task.m` script and set the following properties:
          
     - Task type: results.setup.taskType = 1 or 2
            
       1 = Yes/No task (default)
       
       2 = 2IFC task

     - Staircase type: results.setup.staircaseType = 1 or 2
       
       1 = Constant (default: collects all threshold trials at a constant filter intensity, and may result in additional trials to find the perceptual threshold filter intensity)

       2 = Roving (if set, the filter number can change across the threshold trials to maintain performance at the approximate perceptual threshold, and only the specified number of trials will be collected)

     - Confidence scale: (default example)

       results.setup.confidenceLower = 1

       results.setup.confidenceUpper = 10
  2) Navigate to the main folder (`filter_detection_task`) in matlab, and type `filter_detection_task` into the matlab terminal.
  3) Use the supplied instructions file to explain the task requirements to the participant (see `filter_detection_task/participant_instructions/breathing_task_instructions_{english/german}.doc`).
  4) Follow the matlab prompts to specify whether practice and calibration trials should be run, and the number of trials you aim to complete at threshold (minimum recommendation = 60 trials).
  5) Turn on a low level of pink noise (see *IMPORTANT NOTES* section below) to reduce the influence of any noise cues.
  6) Follow the prompt instructions to complete the task. Only for the very first practice trial (a dummy) is it explicitly explained to the participant whether or not they will receive the load, and then all following practice/calibration/main trials will continue on from each other with no changes in instruction, nor with any feedback given.
  7) Once the task is complete, ask the participant to fill in the provided de-briefing questionnaire (see `filter_detection_task/participant_debriefing/debriefing_detection_{english/german}.doc`). This should help you to determine if any significant strategy switch or learning occurred that would warrant post-hoc subject exclusion.

**IMPORTANT NOTES:**

  1) If at any point you should need to terminate the task, use 'control' + 'c' to exit the loop. All data up to that point will be saved in the specified output file.
  2) If any trials were incorrectly entered by the experimenter, a script is provided that will allow you to overwrite specific trials (see `filter_detection_task_fix.m` using the option 'fix').
  3) If the task is exited early for any reason, a new session can be started (without practice and calibration trials), with the filter number and remaining trials specified by the experimenter. Once the task is complete, the two sets of results can be combined (if the filter number stays the same) using the provided script (`filter_detection_task_fix.m` using the option 'combine'). The original data files will be moved to a new folder and a combined file will be produced in the results directory.
  4) If you would like to run any group analyses using the scripts provided by this toolbox, additional software and code is required (see **IMPORTANT ANALYSIS NOTES** below). This additional software is **NOT** required to run the task itself. 

-----

### TRIAL STRUCTURE

**BASIC STRUCTURE FOR YES/NO TASK (practice, calibration and main trials):**
  1) The participant takes three or more breaths on the baseline breathing system (dummy filter), and then indicates when they are ready by raising their hand. The experimenter (out of view) then removes the dummy and replaces it either with the stacked filters or the dummy again. The participant takes up to three more breaths against the resistance, and then removes the mouthpiece and responds to the question: 'Did you feel any increase in resistance to your breathing?'. The experimenter enters the answer into the command line at the prompt.
  2) The participant then rates on a scale (e.g. 1-10) how confident they are **in their decision** (i.e. not how confident they are that there was a resistance present). On this example scale, 1 = not at all confident, and 10 = extremely confident. The experimenter enters the confidence score at the prompt, and then the script requires a confirmation of all answers to continue to the next trial (any input mistakes can be corrected here). If any mistakes were made, answer the confirmation with 0 (or any other valid key) to repeat the trial prompt. Any mistypings will also trigger a repeat trial presentation on screen for re-input.
  3) Note (repeated): Only for the very first practice trial (a dummy) it is explicitly explained to the participant whether or not they will receive the load, and then all following practice/calibration/main trials will continue on from each other with no changes in instruction, nor with any feedback given.

-----

### TASK OVERVIEW (YES/NO)

**TRIAL ORDER:**

Practice trials, calibration trials and then the main task trials are automatically run (in order) if any/all are specified by the experimenter when prompted.

**PRACTICE TRIALS:**
  1) For the first practice trial, the dummy filter is presented and explicitly explained to the participant. If the participant does not understand or reports an increase in resistance to their breathing, the dummy trial can be repeated (enter 1 for response at the prompt)
  2) It is then explained to the participant that they will not be told whether or not any change in breathing occurs from here onwards. For the second practice trial, 7 filters are presented (large load). If the participant does not report any change in their breathing perception, the practice trial is repeated with an additional filter until perception is reported.

**CALIBRATION TRIALS:**
  1) The number of filters to be presented at each trial during the calibration phase is automatically calculated and prompted at the command line. The aim of the calibration phase is to find approximately the right filter to start the main trials on, although this will be adjusted later if needed. Calibration trials can be chosen to be omitted at the beginning of the task, and manual input of the filter number will be prompted instead.
  2) For the calibration trials, the first trial is always a presentation of the dummy filter. One filter is then added each trial until the participant reports the perception of the filters.
  3) Once a first filter perception has been reported (n), another filter is added for the next trial (n+1). If the participant also reports perception of this additional filter, one filter is taken away (back to n) for a final confirmation trial. If the answer is 'yes' on this confirmation trial, the starting filter number will be n, and if it is 'no', the starting filter number will be n+1. Calibration will finish automatically here.
  4) If the first 'yes' report is followed by a 'no' at n+1, filters will continue to be added until there are two 'yes' responses at ascending filter numbers, before a confirmation trial is conducted and the calibration phase terminates.

**MAIN TASK TRIALS:**
  1) The aim of the main task is to complete a specified number of trials (recommendation >= 60 trials) of a random presentation of either the specified number of filters (via calibration or manual input) and the dummy filter. The target is for participants to be ideally within 65-80% accuracy across the task, with their probable actual accuracy calculated from all trials completed at the current filter level. If participants move outside of this range within the first 30 trials, the script will suggest a filter change, and the trial count will begin again (or continue from the last trial number at the same filter level). Further details as follows:
  2) For each trial, participants will respond whether they perceived it to become harder to breathe or stayed the same (see `breathing_task_instructions` file for detailed instructions), as well as their confidence **IN THEIR DECISION** (i.e. not confidence in the presence of the filters --> this distinction is imperative for metacognition).
  3) After 5 trials, the script will begin to calculate both the observed accuracy (e.g. 7/8 trials correct), as well as the likelihood that the true accuracy lies between 65-80% (if using a constant staircase, or if using a roving staircase then a narrower band of 70-75% is employed as greater regularisation does not come at the cost of additional trials). If the probability that the true accuracy lies within this band falls below 20%, an addition or removal of a filter is automatically suggested. The experimenter is asked for confirmation, or can override the suggestion if necessary.
  4) If the new filter number has not yet been tested in the main trials, the trial count will begin again from 1. If the filter change moves the testing filter number back to that of previous trials, the trial count will pick up again from the last trial at this level.
  5) Once the trial count has reached 30 trials (if using a constant staircase), the script will change to only reporting the performance accuracy every 10 trials. If using a roving staircase, the accuracy will continue to be evaluated at every trial. For the constant staircase >= 30 trials, if the accuracy is within 60-85% a simple confirmation to continue at the current filter number will be asked for. If the accuracy is outside 60-85%, the accuracy of any trials completed at the number of filters directly above and below the current filter number will also be reported. In both instances, the experimenter can choose to continue at the current number of trials or change filter number. If considering changing filters, the experimenter must consider the time required to complete the experiment, and the attention, comfort and motivation of the participant. While completing the full set of trials both at one filter number and within the 60-85% accuracy band is the goal, it must be decided whether the number of completed trials OR the accuracy band is most appropriate to be relaxed when necessary in each experimental setting.
  6) Once trials continue past 10 on the same filter, a graph of both running 10-trial accuracy (if using a constant staircase) and the total cumulative accuracy are plotted for the experimenter to monitor online performance of the participant.
  7) The task will continue until the specified number of trials are reached, either all at one filter level if using a constant staircase, or the total specified trials (regardless of filter number) if using a roving staircase. All results are saved continuously in case of an early termination of the task, and the task can be exited at any time by using 'control' + 'c'.
  8) The task can be restarted (without practice and calibration trials) for any remaining trials if necessary. In this case, name the PPID with a new name (e.g. P003b) otherwise it will throw an error to stop it overriding the original results file. These files can be combined using the provided fixing script (see `filter_detection_task_fix.m`, using the option 'combine'). The original data files will be moved by this script, and a combined file will be produced in the results directory.

-----

### ALTERNATIVE OPTIONS

**YES/NO TASK vs 2IFC TASK CONFIGURATION:**

To run the task as a two-interval forced choice task, first specify this in the 'SET UP THE TASK' section of the main task function (found in `filter_detection_task.m`). The main algorithm and default analysis options will stay the same, but the task instructions will change to presenting the resistance in either the first three-breath interval or the second three breath-interval. Participants will need to choose which interval they thought the resistance was present, and rate their confidence as before.

**STAIRCASE OPTIONS:**

You can decide if you would like to run the task with a 'constant staircase', where all threshold trials are collected at a constant number of filters, or a 'roving staircase', where the filter number can change across the threshold trials. The former option will likely require the collection of additional trials at other filter numbers, but will result in a final set of threshold trials at a constant external load. The roving staircase option will only complete the number of trials set out by the experimenter, but runs the potential risk of not stabilising around the perceptual threshold. To help with this, the accuracy band has been more highly regularised to between 70-75% for this staircase option.

**CONFIDENCE RATING SCALE:**

The default is set between 1 and 10 for the confidence rating scale. In principle you can choose whichever scale you like, however, the larger the rating scale (e.g. 1-100), the more computationally expensive the analysis will become, with large scales requiring exceptionally long fit times. If data has been collected on larger scales, one option would be to 'bin' the confidence data onto a smaller rating scale for feasible metacognition model fit times. This option is not automatically included in this toolbox and would require additional testing.

-----

### ANALYSIS OVERVIEW

This analysis script uses either Brian Maniscalco's single subject analysis or Steve Fleming's Hierarchical Bayesian toolbox (for group analysis) to calculate perceptual decision and metacognitive metrics, specific to data produced by running the `filter_detection_analysis` task.

**IMPORTANT ANALYSIS NOTES:**
  1) The code requires JAGS software to run, which can be found here: <http://mcmc-jags.sourceforge.net/>. The code has been tested on JAGS 3.4.0; there are compatibility issues between matjags and JAGS 4.X.
  2) The code requires the [HMeta-d toolbox](https://github.com/metacoglab/HMeta-d). This needs to be downloaded and placed within the `scripts/` folder.
  3) This script is looking for specific results files that were output from the `filter_detection_task`, which are located in the `results/` folder. The results are expected to contain a results.filterThreshold.xx structure, where values for xx:
     - results.filterThreshold.filterNum: a value to determine the number of filters where trials were performed
     - results.filterThreshold.filters: a vector with 1 meaning filters were presented, 0 when filters were absent (dummy)
     - results.filterThreshold.response: a vector with 1 meaning response was 'yes', 0 when response was 'no'
     - results.filterThreshold.confidence: a vector containing confidence score (1-10) on each trial
     - If your results are formatted differently, refer directly to the original scripts from the [HMeta-d toolbox](https://github.com/metacoglab/HMeta-d).

**TO RUN THIS SCRIPT:**

Type `filter_detection_analysis` into the MATLAB terminal from the main `filter_detection_task` folder, and follow the prompts. For full information on the analysis of this task please see the `filter_detection_analysis.m` file in the `scripts/` folder.

**ANALYSIS OPTIONS OVERVIEW:**

This script allows you to run either:

 - A single subject analysis: A non-Bayesian estimation of meta-d', fitting one subject at a time using a single point maximum likelihood estimation (from the original Maniscalco & Lau 2012 paper, more info can be found here: <http://www.columbia.edu/~bsm2105/type2sdt/archive/index.html)>. HOWEVER: Simulations have shown this analysis to be VERY unstable when estimating meta-d' using 60 trials. It is STRONGLY encouraged to utilise the hierarchical models, or collect many more trials (200+ trials) for each subject to have a more reliable measure of meta-d'.
 - A group mean analysis: A hierarchical Bayesian analysis, whereby all subjects are fit together and information from the group mean is used as prior information for each subject. This hierarchical model helps with much more accurate estimations of meta-d' with small trial numbers, such as using 60 trials per subject.
 - A group difference analysis: A hierarchical Bayesian analysis for independent groups, where each group is fitted separately and then the results are compared. Frequentist statistics (i.e. parametric unpaired T-tests, or non-parametric Wilcoxon signed-rank tests) can be used for all values that are not fitted using hierarchical information, such as d', c, filter number, accuracy and average confidence. As the group values for log(meta-d'/d') are calculated using two separate hierarchical models, group differences are then inferred via comparison of the resulting log(meta-d'/d') distributions, and whether the 95% highest-density interval (HDI) of the difference between the distributions spans zero. The HDI is the shortest possible interval containing 95% of the MCMC samples, and may not be symmetric. 
NOTE: If single subject values for log(meta-d'/d') (or any related meta-d' metric) are required for further analyses that span both groups (such as entry into general linear models), it is recommended to fit all subjects together in one regression model with a regressor denoting group identity.
 - A group regression analysis: A hierarchical Bayesian analysis for one group of subjects, where both the group logMratio and a regression parameter (beta) are simultaneously fit. This model formulation helps with much more accurate estimations of a group beta parameter with small trial numbers (e.g. 60 trials/subject), as individual logMratio values are subject to heavy regularisation towards the group mean. **IMPORTANT NOTE:** This regression parameter beta is fit against logMratio, not Mratio.
 - A session difference analysis (paired group difference): A hierarchical Bayesian analysis, whereby two sessions / measures from the same participants are fitted in a single model using a multivariate normal distribution. This distribution allows for the non-independence between sessions for each participant. NOTE: Participants must be listed in the same order for analysis, and must have data for both sessions / each measure.

**IMPORTANT ANALYSIS NOTES:**

 - The code requires JAGS software to run, which can be found here: <http://mcmc-jags.sourceforge.net/>. The code has been tested on JAGS 3.4.0; there are compatibility issues between matjags and JAGS 4.X.
 - This script is looking for specific results files that were output from the `filter_detection_task`, which are located in the `results/` folder. For full information on the requirements of this analysis please see the `filter_detection_analysis.m` file in the `scripts/` folder.

**KEY ANALYSIS OUTPUT MEASURES:**

 - filterNum: Number of filters. Less filters means more sensitive to changes in breathing.
 - d1: d prime, the discriminability between filter and dummy trials. Larger d' means more discriminability at specific filter number
 - c1: Decision criterion, or where the decision boundary exists. Negative criterion values indicate bias towards 'yes', positive values indicate a bias towards a 'no' response.
 - meta_d: A measure of metacognition, or 'type 2' sensitivity. This reflects how much information, in signal-to-noise units, is available for metacognition (see Maniscalco & Lau, 2012). **NB: ISSUES WITH THIS ESTIMATION (AND ALL RELATED ESTIMATIONS) IF USING SINGLE SUBJECT ANALYSES, AS DESCRIBED ABOVE**.
 - Mratio: Meta-d' to d' ratio, as a measure of metacognitive 'efficiency' (see Rouault et al., 2018) --> how much information is lost from discriminability to meta-cognition)
 - log_Mratio: log(meta_d/d1), reported in papers to help with normalisation of data (see Rouault et al., 2018).
 - avgConfidence: A second measure of metacognition, thought to be independent from meta-d'. NB: Average confidence should only be compared between two groups if there is no difference in d' between the groups (i.e. task difficulty was comparable --> if it is not, this measure will need to be corrected for differences in d')

**ADDITIONAL GROUP / SESSION DIFFERENCE OUTPUT VALUES:**

If the analysis is a two-group or two-session (paired) difference, the following will also be calculated for the non-hierarchical measures (xx = filterNum, d1, c1 and avgConfidence): (test = groupDiff or sessionDiff)
  1) analysis.test.xx.h = Results of null hypothesis test, where h = 1 for rejection of the null, and h = 0 for no rejection.
  2) analysis.test.xx.p = The p-value for the statistical test.
  3) analysis.test.xx.stats = Further statistics (such as tstats, number of samples, degrees of freedom) associated with the test.
  4) analysis.test.xx.ci = Confidence interval of the difference, calculated only when T-test is specified.

The highest density interval (HDI) for the difference in log(meta-d'/d') between the groups / sessions will be calculated and recorded, as frequentist statistics cannot be used here. A summary Figure for each of these metrics will be created and saved in the `analysis/` folder.