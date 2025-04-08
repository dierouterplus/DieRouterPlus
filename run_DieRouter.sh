#!/bin/bash

Scripts_Dir="./Source/Baseline_Scripts"
ContTDMOpt_Script_Dir="./Source/ContTDMOpt_Scripts"

N_Pins_Gap_Factor="60"
# Routing_Critical_Net : "Dijkstra"/"MST"/"SMT"
Routing_Critical_Net="MST"
Routing_Non_Critical_Net="Dijkstra"


Two_Stage_Reroute_Method="run_two_stage_reroute_sink"
Edge_Criticality_Metric="Max"
Patience="2"

#Conti_Sol
significance_threshold="1"

#Legalization Method: DP-Legalization, DP-Legalization-Baseline, DP-Legalization-Unbalance
legalization_token="DP-Legalization-Baseline"


for i in {1..5}
do
    testcase="testcase$i"

    echo "Processing $testcase ..."

    cd $Scripts_Dir || { echo "Cannot switch to $Scripts_Dir. Skip $testcase."; continue; }

    # Run initial routing
    python run_hybrid_initial_routing.py \
        --testcase_dir "../../Data/$testcase" \
        --critical_net ${Routing_Critical_Net} \
        --non_critical_net ${Routing_Non_Critical_Net} \
        --n_pins_gap_factor ${N_Pins_Gap_Factor}

    if [ $? -ne 0 ]; then
        echo "Error in performing run_hybrid_initial_routing on $testcase"
        exit 1
    fi

    # Run two stage reroute
    python ${Two_Stage_Reroute_Method}.py \
    --up_dir ../../Res/$testcase \
    --identifier_0_n_pins_gap_factor ${N_Pins_Gap_Factor} \
    --identifier_0_token Hybrid-Initial-Routing-${Routing_Critical_Net}-${Routing_Non_Critical_Net} \
    --edge_criticality_metric ${Edge_Criticality_Metric} \
    --patience ${Patience}


    if [ $? -ne 0 ]; then
        echo Error in performing $Two_Stage_Reroute_Method on $testcase
        exit 1
    fi

    # Solve Continuous TDM Ratios
    identifier_1_token=Two-Stage-Reroute-Sink-${Edge_Criticality_Metric}-zero


    python run_conti_tdm_solver.py \
    --up_dir ../../Res/$testcase \
    --significance_threshold ${significance_threshold} \
    --identifier_0_n_pins_gap_factor ${N_Pins_Gap_Factor} \
    --identifier_0_token Hybrid-Initial-Routing-${Routing_Critical_Net}-${Routing_Non_Critical_Net} \
    --identifier_1_token ${identifier_1_token}


    python run_legalization.py \
    --up_dir ../../Res/$testcase \
    --enable_multiprocessing \
    --n_process 10 \
    --token ${legalization_token} \
    --identifier_0_n_pins_gap_factor ${N_Pins_Gap_Factor} \
    --identifier_0_token Hybrid-Initial-Routing-${Routing_Critical_Net}-${Routing_Non_Critical_Net} \
    --identifier_1_token ${identifier_1_token} \
    --identifier_2_token "Conti-TDM-Ratio-Baseline"

    echo "Finish $testcase"
    cd ../..
    echo "-----------------------------"
done

echo "Finish"

