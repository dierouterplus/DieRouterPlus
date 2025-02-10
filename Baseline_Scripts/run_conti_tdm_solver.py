import argparse
import json
import pickle
import sys
sys.path.append('../')

from Baseline.ContiTDMSolver.solver import Solver
from Baseline.ContiTDMSolver.dynamic_solver import DynamicSolver

from pathlib import Path
import time
import os
from utils.parse_result import check_result_exists, get_previous_step_output_path
from Baseline.baseline_utils.process_raw_data import build_weighted_routing_resource_network_and_nets
from Baseline.HybridInitialRouting.hybrid_initial_routing import HybridInitialRouting
from utils.save_result import save_experiment_results

def parse_args():
    parser = argparse.ArgumentParser(description="Continuous TDM Solver")

    parser.add_argument(
        '--up_dir',
        type=str,
        help='Directory for results.'
    )

    parser.add_argument(
        '--significance_threshold',
        type=float,
        default=1e-3,
        help="Threshold for significance (default: 1e-3)"
    )
    parser.add_argument(
        '--stag_rounds_threshold',
        type=int,
        default=10,
        help="Threshold for stagnation rounds (default: 10)"
    )
    parser.add_argument(
        '--dynamic_solve',
        type=bool,
        default=False,
        help="Enable or disable dynamic solve (default: False)"
    )

    # identifier 参数
    parser.add_argument(
        '--identifier_0_n_pins_gap_factor',
        type=int,
        help='Gap factor for the number of pins in identifier 0.'
    )

    parser.add_argument(
        '--identifier_0_token',
        type=str,
        help='Token for identifier 0.'
    )

    parser.add_argument(
        '--identifier_1_token',
        type=str,
        help='Token for identifier 1.'
    )

    parser.add_argument(
        '--token',
        type=str,
        default='Conti-TDM-Ratio-Baseline',
        help='Token name for the routing process.'
    )

    return parser.parse_args()



if __name__ == "__main__":
    """
    Example:
    kwargs = {
        'up_dir': '../Res/Baseline/testcase10',
        'significance_threshold': 1e-3,
        'stag_rounds_threshold': 10,
        'dynamic_solve': False,
        'identifier': {
            '0': {
                'n_pins_gap_factor': 60,
                'token': 'Hybrid-Initial-Routing-SMT-Dijkstra'
            },
            '1': {
                'token': 'Two-Stage-Reroute-Sink-Max-zero'
            }
        },
        'token': 'Conti-TDM-Ratio-Baseline'
    }
    """

    args = parse_args()
    print(json.dumps(vars(args), indent=4))

    identifier = {
        '0': {
            'n_pins_gap_factor': args.identifier_0_n_pins_gap_factor,
            'token': args.identifier_0_token
        },
        '1': {
            'token': args.identifier_1_token
        }
    }

    kwargs = {
        'up_dir': args.up_dir,
        'significance_threshold': args.significance_threshold,
        'stag_rounds_threshold': args.stag_rounds_threshold,
        'dynamic_solve': args.dynamic_solve,
        'identifier': identifier,
        'token': args.token
    }


    res_dir = get_previous_step_output_path(kwargs['up_dir'], kwargs['identifier'])
    is_result_present, _ = check_result_exists(res_dir, kwargs)

    if is_result_present:
        print(f"Result already presents in {_}")
    else:
        two_stage_reroute_res_path = f"{res_dir}/two_stage_reroute_res.pkl"
        with open(two_stage_reroute_res_path, "rb") as f:
            nets, weighted_routing_resource_network, _ = pickle.load(f)

        start = time.time()
        if kwargs['dynamic_solve']:
            solver = DynamicSolver(nets, weighted_routing_resource_network,
                                   sig_thresh=kwargs['significance_threshold'],
                                   stag_rounds_thresh=kwargs['stag_rounds_threshold'])
        else:
            solver = Solver(nets, weighted_routing_resource_network,
                            sig_thresh=kwargs['significance_threshold'],
                            stag_rounds_thresh=kwargs['stag_rounds_threshold'])

        solver.perf_init_assign_and_refine()
        end = time.time()

        print('Solve Continuous TDM Ratios took {} seconds'.format(end - start))

        save_experiment_results([solver.nets,
                                 solver.weighted_routing_resource_net,
                                 solver.best_dir_tdm_edge_path_tdm_ratio_matrix,
                                 solver.directed_tdm_edge_row_idx_map],
                                kwargs, res_dir, 'conti_tdm_ratio.pkl')












