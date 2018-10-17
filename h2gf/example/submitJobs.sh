#!/bin/bash
#

for i in {1..10}; do for k in {1..6}; do for j in {1..12};do bsub -W 71:55 -o o_h2gf_1000_eta$k_m$i_config$j -e e_h2gf_1000_eta$k_m$i_config$j matlab -singleCompThread -r "h2gf_demo_srl2($i,1000,$k,$j)"; done; done; done;
for i in {1..10}; do for k in {1..6}; do for j in {1..12};do bsub -W 71:55 -o o_h2gf_3000_eta$k_m$i_config$j -e e_h2gf_3000_eta$k_m$i_config$j matlab -singleCompThread -r "h2gf_demo_srl2($i,3000,$k,$j)"; done; done; done;
for i in {1..10}; do for k in {1..6}; do for j in {1..12};do bsub -W 71:55 -o o_h2gf_4000_eta$k_m$i_config$j -e e_h2gf_4000_eta$k_m$i_config$j matlab -singleCompThread -r "h2gf_demo_srl2($i,4000,$k,$j)"; done; done; done;
for i in {1..10}; do for k in {1..6}; do for j in {1..12};do bsub -W 71:55 -o o_h2gf_5000_eta$k_m$i_config$j -e e_h2gf_5000_eta$k_m$i_config$j matlab -singleCompThread -r "h2gf_demo_srl2($i,5000,$k,$j)"; done; done; done;


for k in {1..12}; do for j in {1..12};do bsub -W 3:55 -o o_srl1_boxplot_h2gf_1000_eta$k_config$j -e e_srl1_boxplot_h2gf_1000_eta$k_config$j matlab -singleCompThread -r "h2gf_demo_srl2_summary(1000,$k,$j)"; done; done;
for k in {1..12}; do for j in {1..12};do bsub -W 3:55 -o o_srl1_boxplot_h2gf_3000_eta$k_config$j -e e_srl1_boxplot_h2gf_3000_eta$k_config$j matlab -singleCompThread -r "h2gf_demo_srl2_summary(3000,$k,$j)"; done; done;
for k in {1..12}; do for j in {1..12};do bsub -W 3:55 -o o_srl1_boxplot_h2gf_4000_eta$k_config$j -e e_srl1_boxplot_h2gf_4000_eta$k_config$j matlab -singleCompThread -r "h2gf_demo_srl2_summary(4000,$k,$j)"; done; done;
for k in {1..12}; do for j in {1..12};do bsub -W 3:55 -o o_srl1_boxplot_h2gf_5000_eta$k_config$j -e e_srl1_boxplot_h2gf_5000_eta$k_config$j matlab -singleCompThread -r "h2gf_demo_srl2_summary(5000,$k,$j)"; done; done;