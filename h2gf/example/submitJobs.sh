#!/bin/bash
#

#run h2gf srl2
for i in {1..12}; do for k in {1..6}; do for j in {1..12};do bsub -b 04:00 -W 3:55 -o o_h2gf_1000_eta$k_m$i_config$j -e e_h2gf_1000_eta$k_m$i_config$j matlab -singleCompThread -r "h2gf_demo_srl2($i,1000,$k,$j)"; done; done; done;
for i in {1..12}; do for k in {1..6}; do for j in {1..12};do bsub -b 02:00 -W 7:55 -o o_h2gf_3000_eta$k_m$i_config$j -e e_h2gf_3000_eta$k_m$i_config$j matlab -singleCompThread -r "h2gf_demo_srl2($i,3000,$k,$j)"; done; done; done;
for i in {1..12}; do for k in {1..6}; do for j in {1..12};do bsub -b 23:30 -W 7:55 -o o_h2gf_4000_eta$k_m$i_config$j -e e_h2gf_4000_eta$k_m$i_config$j matlab -singleCompThread -r "h2gf_demo_srl2($i,4000,$k,$j)"; done; done; done;
for i in {1..12}; do for k in {1..6}; do for j in {1..12};do bsub -b 22:00 -W 7:55 -o o_h2gf_5000_eta$k_m$i_config$j -e e_h2gf_5000_eta$k_m$i_config$j matlab -singleCompThread -r "h2gf_demo_srl2($i,5000,$k,$j)"; done; done; done;

#run h2gf srl1
for i in {1..12}; do for k in {1..6}; do for j in {1..12};do bsub -b 21:00 -W 3:55 -o o_srl1_h2gf_1000_eta$k_m$i_config$j -e e_srl1_h2gf_1000_eta$k_m$i_config$j matlab -singleCompThread -r "h2gf_demo_srl1($i,1000,$k,$j)"; done; done; done;
for i in {1..12}; do for k in {1..6}; do for j in {1..12};do bsub -b 19:00 -W 7:55 -o o_srl1_h2gf_3000_eta$k_m$i_config$j -e e_srl1_h2gf_3000_eta$k_m$i_config$j matlab -singleCompThread -r "h2gf_demo_srl1($i,3000,$k,$j)"; done; done; done;
for i in {1..12}; do for k in {1..6}; do for j in {1..12};do bsub -b 18:30 -W 7:55 -o o_srl1_h2gf_4000_eta$k_m$i_config$j -e e_srl1_h2gf_4000_eta$k_m$i_config$j matlab -singleCompThread -r "h2gf_demo_srl1($i,4000,$k,$j)"; done; done; done;
for i in {1..12}; do for k in {1..6}; do for j in {1..12};do bsub -W 7:55 -o o_srl1_h2gf_5000_eta$k_m$i_config$j -e e_srl1_h2gf_5000_eta$k_m$i_config$j matlab -singleCompThread -r "h2gf_demo_srl1($i,5000,$k,$j)"; done; done; done;

#run h2gf_rw srl2
for i in {1..12}; do for k in {1..6}; do bsub -b 02:00 -W 3:55 -o o_h2gf_1000_eta$k_m$i_rw -e e_h2gf_1000_eta$k_m$i_rw matlab -singleCompThread -r "h2gf_demo_srl2_rw($i,1000,$k)"; done; done; 
for i in {1..12}; do for k in {1..6}; do bsub -b 23:00 -W 7:55 -o o_h2gf_3000_eta$k_m$i_rw -e e_h2gf_3000_eta$k_m$i_rw matlab -singleCompThread -r "h2gf_demo_srl2_rw($i,3000,$k)"; done; done; 
for i in {1..12}; do for k in {1..6}; do bsub -b 21:30 -W 7:55 -o o_h2gf_4000_eta$k_m$i_rw -e e_h2gf_4000_eta$k_m$i_rw matlab -singleCompThread -r "h2gf_demo_srl2_rw($i,4000,$k)"; done; done; 
for i in {1..12}; do for k in {1..6}; do bsub -b 19:00 -W 7:55 -o o_h2gf_5000_eta$k_m$i_rw -e e_h2gf_5000_eta$k_m$i_rw matlab -singleCompThread -r "h2gf_demo_srl2_rw($i,5000,$k)"; done; done; 

#run h2gf_rw srl1
for i in {1..12}; do for k in {1..6}; do bsub -W 03:55 -o o_srl1_h2gf_1000_eta$k_m$i_rw -e e_srl1_h2gf_1000_eta$k_m$i_rw matlab -singleCompThread -r "h2gf_demo_srl1_rw($i,1000,$k)"; done; done; 
for i in {1..12}; do for k in {1..6}; do bsub -b 16:00 -W 7:55 -o o_srl1_h2gf_3000_eta$k_m$i_rw -e e_srl1_h2gf_3000_eta$k_m$i_rw matlab -singleCompThread -r "h2gf_demo_srl1_rw($i,3000,$k)"; done; done; 
for i in {1..12}; do for k in {1..6}; do bsub -b 14:30 -W 7:55 -o o_srl1_h2gf_4000_eta$k_m$i_rw -e e_srl1_h2gf_4000_eta$k_m$i_rw matlab -singleCompThread -r "h2gf_demo_srl1_rw($i,4000,$k)"; done; done; 
for i in {1..12}; do for k in {1..6}; do bsub -W 7:55 -o o_srl1_h2gf_5000_eta$k_m$i_rw -e e_srl1_h2gf_5000_eta$k_m$i_rw matlab -singleCompThread -r "h2gf_demo_srl1_rw($i,5000,$k)"; done; done; 

#plot boxplots srl2
for k in {1..12}; do for j in {1..12};do bsub -W 0:55 -o o_srl2_boxplot_h2gf_1000_eta$k_config$j -e e_srl2_boxplot_h2gf_1000_eta$k_config$j matlab -singleCompThread -r "h2gf_demo_srl2_summary(1000,$k,$j)"; done; done;
for k in {1..12}; do for j in {1..12};do bsub -W 0:55 -o o_srl2_boxplot_h2gf_3000_eta$k_config$j -e e_srl2_boxplot_h2gf_3000_eta$k_config$j matlab -singleCompThread -r "h2gf_demo_srl2_summary(3000,$k,$j)"; done; done;
for k in {1..12}; do for j in {1..12};do bsub -W 0:55 -o o_srl2_boxplot_h2gf_4000_eta$k_config$j -e e_srl2_boxplot_h2gf_4000_eta$k_config$j matlab -singleCompThread -r "h2gf_demo_srl2_summary(4000,$k,$j)"; done; done;
for k in {1..12}; do for j in {1..12};do bsub -W 0:55 -o o_srl2_boxplot_h2gf_5000_eta$k_config$j -e e_srl2_boxplot_h2gf_5000_eta$k_config$j matlab -singleCompThread -r "h2gf_demo_srl2_summary(5000,$k,$j)"; done; done;

#plot boxplots srl1
for k in {1..12}; do for j in {1..12};do bsub -W 0:55 -o o_srl1_boxplot_h2gf_1000_eta$k_config$j -e e_srl1_boxplot_h2gf_1000_eta$k_config$j matlab -singleCompThread -r "h2gf_demo_srl1_summary(1000,$k,$j)"; done; done;
for k in {1..12}; do for j in {1..12};do bsub -W 0:55 -o o_srl1_boxplot_h2gf_3000_eta$k_config$j -e e_srl1_boxplot_h2gf_3000_eta$k_config$j matlab -singleCompThread -r "h2gf_demo_srl1_summary(3000,$k,$j)"; done; done;
for k in {1..12}; do for j in {1..12};do bsub -W 0:55 -o o_srl1_boxplot_h2gf_4000_eta$k_config$j -e e_srl1_boxplot_h2gf_4000_eta$k_config$j matlab -singleCompThread -r "h2gf_demo_srl1_summary(4000,$k,$j)"; done; done;
for k in {1..12}; do for j in {1..12};do bsub -W 0:55 -o o_srl1_boxplot_h2gf_5000_eta$k_config$j -e e_srl1_boxplot_h2gf_5000_eta$k_config$j matlab -singleCompThread -r "h2gf_demo_srl1_summary(5000,$k,$j)"; done; done;

#plot boxplots rw srl2
for k in {1..12}; do bsub -W 0:55 -o o_srl2_boxplot_h2gf_1000_eta$k_rw -e e_srl2_boxplot_h2gf_1000_eta$k_rw matlab -singleCompThread -r "h2gf_demo_srl2_summary_rw(1000,$k)"; done; 
for k in {1..12}; do bsub -W 0:55 -o o_srl2_boxplot_h2gf_3000_eta$k_rw -e e_srl2_boxplot_h2gf_3000_eta$k_rw matlab -singleCompThread -r "h2gf_demo_srl2_summary_rw(3000,$k)"; done;
for k in {1..12}; do bsub -W 0:55 -o o_srl2_boxplot_h2gf_4000_eta$k_rw -e e_srl2_boxplot_h2gf_4000_eta$k_rw matlab -singleCompThread -r "h2gf_demo_srl2_summary_rw(4000,$k)"; done;
for k in {1..12}; do bsub -W 0:55 -o o_srl2_boxplot_h2gf_5000_eta$k_rw -e e_srl2_boxplot_h2gf_5000_eta$k_rw matlab -singleCompThread -r "h2gf_demo_srl2_summary_rw(5000,$k)"; done;

#plot boxplots rw srl1
for k in {1..12}; do bsub -W 0:55 -o o_srl1_boxplot_h2gf_1000_eta$k_rw -e e_srl1_boxplot_h2gf_1000_eta$k_rw matlab -singleCompThread -r "h2gf_demo_srl1_summary_rw(1000,$k)"; done;
for k in {1..12}; do bsub -W 0:55 -o o_srl1_boxplot_h2gf_3000_eta$k_rw -e e_srl1_boxplot_h2gf_3000_eta$k_rw matlab -singleCompThread -r "h2gf_demo_srl1_summary_rw(3000,$k)"; done;
for k in {1..12}; do bsub -W 0:55 -o o_srl1_boxplot_h2gf_4000_eta$k_rw -e e_srl1_boxplot_h2gf_4000_eta$k_rw matlab -singleCompThread -r "h2gf_demo_srl1_summary_rw(4000,$k)"; done;
for k in {1..12}; do bsub -W 0:55 -o o_srl1_boxplot_h2gf_5000_eta$k_rw -e e_srl1_boxplot_h2gf_5000_eta$k_rw matlab -singleCompThread -r "h2gf_demo_srl1_summary_rw(5000,$k)"; done;

#combine boxplots
for c in {1..8}; do for j in {1..12}; do bsub -W 0:55 -o o_srl2_linkfigures_1000_$c_config$j -e e_srl1_linkfigures_1000_$c_config$j matlab -singleCompThread -r "h2gf_demo_srl2_link_figures(1000,$j,$c)"; done; done;
for c in {1..8}; do for j in {1..12}; do bsub -W 0:55 -o o_srl2_linkfigures_3000_$c_config$j -e e_srl1_linkfigures_3000_$c_config$j matlab -singleCompThread -r "h2gf_demo_srl2_link_figures(3000,$j,$c)"; done; done;
for c in {1..8}; do for j in {1..12}; do bsub -W 0:55 -o o_srl2_linkfigures_4000_$c_config$j -e e_srl1_linkfigures_4000_$c_config$j matlab -singleCompThread -r "h2gf_demo_srl2_link_figures(4000,$j,$c)"; done; done;
for c in {1..8}; do for j in {1..12}; do bsub -W 0:55 -o o_srl2_linkfigures_5000_$c_config$j -e e_srl1_linkfigures_5000_$c_config$j matlab -singleCompThread -r "h2gf_demo_srl2_link_figures(5000,$j,$c)"; done; done;
for c in {1..8}; do for j in {1..12}; do bsub -W 0:55 -o o_srl1_linkfigures_1000_$c_config$j -e e_srl1_linkfigures_1000_$c_config$j matlab -singleCompThread -r "h2gf_demo_srl1_link_figures(1000,$j,$c)"; done; done;
for c in {1..8}; do for j in {1..12}; do bsub -W 0:55 -o o_srl1_linkfigures_3000_$c_config$j -e e_srl1_linkfigures_3000_$c_config$j matlab -singleCompThread -r "h2gf_demo_srl1_link_figures(3000,$j,$c)"; done; done;
for c in {1..8}; do for j in {1..12}; do bsub -W 0:55 -o o_srl1_linkfigures_4000_$c_config$j -e e_srl1_linkfigures_4000_$c_config$j matlab -singleCompThread -r "h2gf_demo_srl1_link_figures(4000,$j,$c)"; done; done;
for c in {1..8}; do for j in {1..12}; do bsub -W 0:55 -o o_srl1_linkfigures_5000_$c_config$j -e e_srl1_linkfigures_5000_$c_config$j matlab -singleCompThread -r "h2gf_demo_srl1_link_figures(5000,$j,$c)"; done; done;

#plot individual trajectories:
for c in {1..6}; do for j in {1..13}; do bsub -W 0:55 -o o_srl2_plotTraj_h2gf_1000_eta$c_config$j -e e_srl2_plotTraj_h2gf_1000_eta$c_config$j matlab -singleCompThread -r "h2gf_demo_srl2_plot_data(1000,$c,$j,1)"; done; done;
for c in {1..6}; do for j in {1..13}; do bsub -W 0:55 -o o_srl2_plotTraj_h2gf_3000_eta$c_config$j -e e_srl2_plotTraj_h2gf_3000_eta$c_config$j matlab -singleCompThread -r "h2gf_demo_srl2_plot_data(3000,$c,$j,1)"; done; done;
for c in {1..6}; do for j in {1..13}; do bsub -W 0:55 -o o_srl2_plotTraj_h2gf_4000_eta$c_config$j -e e_srl2_plotTraj_h2gf_4000_eta$c_config$j matlab -singleCompThread -r "h2gf_demo_srl2_plot_data(4000,$c,$j,1)"; done; done;
for c in {1..6}; do for j in {1..13}; do bsub -W 0:55 -o o_srl2_plotTraj_h2gf_5000_eta$c_config$j -e e_srl2_plotTraj_h2gf_5000_eta$c_config$j matlab -singleCompThread -r "h2gf_demo_srl2_plot_data(5000,$c,$j,1)"; done; done;
for c in {1..6}; do for j in {1..13}; do bsub -W 0:55 -o o_srl1_plotTraj_h2gf_1000_eta$c_config$j -e e_srl1_plotTraj_h2gf_1000_eta$c_config$j matlab -singleCompThread -r "h2gf_demo_srl1_plot_data(1000,$c,$j,1)"; done; done;
for c in {1..6}; do for j in {1..13}; do bsub -W 0:55 -o o_srl1_plotTraj_h2gf_3000_eta$c_config$j -e e_srl1_plotTraj_h2gf_3000_eta$c_config$j matlab -singleCompThread -r "h2gf_demo_srl1_plot_data(3000,$c,$j,1)"; done; done;
for c in {1..6}; do for j in {1..13}; do bsub -W 0:55 -o o_srl1_plotTraj_h2gf_4000_eta$c_config$j -e e_srl1_plotTraj_h2gf_4000_eta$c_config$j matlab -singleCompThread -r "h2gf_demo_srl1_plot_data(4000,$c,$j,1)"; done; done;
for c in {1..6}; do for j in {1..13}; do bsub -W 0:55 -o o_srl1_plotTraj_h2gf_5000_eta$c_config$j -e e_srl1_plotTraj_h2gf_5000_eta$c_config$j matlab -singleCompThread -r "h2gf_demo_srl1_plot_data(5000,$c,$j,1)"; done; done;

#plot inference:
for c in {1..6}; do for j in {1..13}; do bsub -W 0:55 -o o_srl2_plotInf_h2gf_eta$c_config$j -e e_srl2_plotInf_h2gf_eta$c_config$j matlab -singleCompThread -r "h2gf_demo_srl2_plot_inference($c,$j,1)"; done; done;
for c in {1..6}; do for j in {1..13}; do bsub -W 0:55 -o o_srl1_plotInf_h2gf_eta$c_config$j -e e_srl1_plotInf_h2gf_eta$c_config$j matlab -singleCompThread -r "h2gf_demo_srl1_plot_inference($c,$j,1)"; done; done;

HGF v5.2
#run hgf srl2
for i in {1..13}; do bsub -W 3:55 -o o_hgfsrl2_config_$i -e e_hgfsrl1_config_$i matlab -singleCompThread -r "hgfv5_2_srl2($i)"; done; 
for i in {1..13}; do bsub -W 3:55 -o o_hgfsrl2_summary_config_$i -e e_hgfsrl1_summary_config_$i matlab -singleCompThread -r "hgfv5_2_srl2_summary($i)"; done; 

#run hgf srl1
for i in {1..13}; do bsub -W 3:55 -o o_hgfsrl1_config_$i -e e_hgfsrl1_config_$i matlab -singleCompThread -r "hgfv5_2_srl1($i)"; done; 
for i in {1..13}; do bsub -W 3:55 -o o_hgfsrl1_summary_config_$i -e e_hgfsrl1_summary_config_$i matlab -singleCompThread -r "hgfv5_2_srl1_summary($i)"; done;

#plot stats for h2gf and hgf
for k in {1..12}; do bsub -W 0:55 -o o_srl2_stats_config$k -e e_srl2_stats_config$k matlab -singleCompThread -r "h2gf_demo_srl2_stats_plot($k)"; done;
for k in {1..12}; do bsub -W 0:55 -o o_srl1_stats_config$k -e e_srl1_stats_config$k matlab -singleCompThread -r "h2gf_demo_srl2_stats_plot($k)"; done;
