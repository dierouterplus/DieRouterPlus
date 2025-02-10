import pickle
import sys
sys.path.append('../../')

from Source.Baseline.Discretization.incremental_dp_legal_load_balance import BalancedIncrementalDPLegal
from Source.Baseline.Discretization.incremental_dp_legal import IncrementalDPLegal
from Source.Baseline.Discretization.all_resources_dp_legal import AllResourcesDPLegal

from Source.utils.parse_result import check_result_exists, get_previous_step_output_path
from Source.utils.save_result import save_experiment_results

import argparse
import json
import multiprocessing


def parse_args():
    parser = argparse.ArgumentParser(description="Legalization Configuration")

    parser.add_argument(
        '--up_dir',
        type=str,
        help='Directory for results.'
    )

    parser.add_argument(
        '--enable_multiprocessing',
        action='store_true',
        help='Enable multiprocessing.'
    )

    parser.add_argument(
        '--n_process',
        type=int,
        default=10,
        help='Number of processes. Set to -1 for auto detection based on CPU count.'
    )

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
        '--identifier_2_token',
        type=str,
        help='Token for identifier 2.'
    )

    parser.add_argument(
        '--token',
        type=str,
        default='DP-Legalization',
        choices=['DP-Legalization', 'DP-Legalization-Baseline', 'DP-Legalization-Unbalance'],
        help='Token name for the routing process. Choices are DP-Legalization, '
             'DP-Legalization-Baseline, and DP-Legalization-unbalance.'
    )

    return parser.parse_args()



if __name__ == "__main__":
    """
    Example
    kwargs = {
        'up_dir': '../Res/Baseline/testcase10',
        'enable_multiprocessing': True,
        'n_process': 10, # -1 means invoking cpu_count() to set the number of processes.
        'identifier': {
            '0': {
                'n_pins_gap_factor': 60,
                'token': 'Hybrid-Initial-Routing-SMT-Dijkstra'
            },
            '1': {
                'token': 'Two-Stage-Reroute-Sink-Max-zero'
            },
            # '2': {
            #     'significance_threshold': 1e-3,
            #     'stag_rounds_threshold': 10,
            #     'dynamic_solve': False,
            #     'token': 'Conti-TDM-Ratio'
            # }
            '2': {
                'token': 'Conti-TDM-Ratio-Mosek'
            }
        },
        'token': 'DP-Legalization'
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
        },
        '2': {
            'token': args.identifier_2_token
        }
    }

    if args.enable_multiprocessing:
        n_process = args.n_process if args.n_process != -1 else multiprocessing.cpu_count()
    else:
        n_process = 1

    kwargs = {
        'up_dir': args.up_dir,
        'enable_multiprocessing': args.enable_multiprocessing,
        'n_process': n_process,
        'identifier': identifier,
        'token': args.token
    }


    res_dir = get_previous_step_output_path(kwargs['up_dir'], kwargs['identifier'])
    is_result_present, _ = check_result_exists(res_dir, kwargs)

    if is_result_present:
        print(f"Result already presents in {_}")
    else:
        conti_tdm_ratio_res_path = f"{res_dir}/conti_tdm_ratio.pkl"
        with open(conti_tdm_ratio_res_path, "rb") as f:
            nets, weighted_routing_resource_net, dir_tdm_edge_path_tdm_ratio_matrix, directed_tdm_edge_row_idx_map = pickle.load(f)


        dp_legalizer = None
        if args.token == 'DP-Legalization':
            dp_legalizer = BalancedIncrementalDPLegal(nets, weighted_routing_resource_net, dir_tdm_edge_path_tdm_ratio_matrix,
                                   directed_tdm_edge_row_idx_map, kwargs['enable_multiprocessing'], kwargs['n_process'])
        elif args.token == 'DP-Legalization-Baseline':
            dp_legalizer = AllResourcesDPLegal(nets, weighted_routing_resource_net, dir_tdm_edge_path_tdm_ratio_matrix,
                                   directed_tdm_edge_row_idx_map, kwargs['enable_multiprocessing'], kwargs['n_process'])
        elif args.token == 'DP-Legalization-Unbalance':
            dp_legalizer = IncrementalDPLegal(nets, weighted_routing_resource_net, dir_tdm_edge_path_tdm_ratio_matrix,
                                   directed_tdm_edge_row_idx_map, kwargs['enable_multiprocessing'], kwargs['n_process'])

        dp_legalizer.perform_dp_legal()

        save_experiment_results([dp_legalizer.nets,
                                 dp_legalizer.weighted_routing_resource_net,
                                 dp_legalizer.dir_tdm_edge_path_tdm_ratio_matrix,
                                 dp_legalizer.directed_tdm_edge_to_row_idx],
                                kwargs, res_dir, 'legalized_tdm_ratio.pkl')












