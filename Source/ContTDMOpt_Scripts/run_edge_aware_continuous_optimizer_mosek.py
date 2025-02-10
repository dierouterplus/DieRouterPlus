import pickle
import sys

import numpy as np

sys.path.append('../../')

from Source.ContTDMOpt.convex_edge_aware_optimizer_mosek import ConvexEdgeAwareOptimizer

from Source.utils.parse_result import check_result_exists, get_previous_step_output_path
from Source.utils.save_result import save_experiment_results
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Edge Aware Continuous Optimizer Configuration")

    parser.add_argument(
        '--up_dir',
        type=str,
        help='Directory for results.'
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
        default='Conti-TDM-Ratio-Mosek',
        help='Token name for the routing process.'
    )

    return parser.parse_args()



if __name__ == "__main__":
    """
    Example:
    kwargs = {
        'up_dir': '../Res/Baseline/testcase10',
        'identifier': {
            '0': {
                'n_pins_gap_factor': 60,
                'token': 'Hybrid-Initial-Routing-SMT-Dijkstra'
            },
            '1': {
                'token': 'Two-Stage-Reroute-Sink-Max-zero'
            }
        },
        'token': 'Conti-TDM-Ratio-Mosek'
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



        solver = ConvexEdgeAwareOptimizer(nets, weighted_routing_resource_network)

        """
        solved_var_matrix: NumpyArray, (n_tdm_edges, n_nets)
        """
        solved_var_matrix, primal_obj_val, primal_sol_status, path_bias = solver.solve()

        print('System delay after Continuous Optimization is {}'.format(primal_obj_val))

        """
        Construct data structures to facilitate use in Baseline.Legal.
        [1] directed_tdm_edge_row_idx_map: dict(directed_tdm_edge -> row_idx)
        [2] dir_tdm_edge_path_tdm_ratio_matrix: NumpyArray, (2 * n_tdm_edges, n_paths)
        [3] For each net, initialize net.directed_tdm_edge_to_sink_idx_list.
        """
        directed_tdm_edge_row_idx_map = {}
        row_idx = 0
        for tdm_edge in solver.tdm_edge_row_idx_map:
            directed_tdm_edge_row_idx_map[(tdm_edge[0], tdm_edge[1])] = row_idx
            directed_tdm_edge_row_idx_map[(tdm_edge[1], tdm_edge[0])] = row_idx + 1
            row_idx += 2


        dir_tdm_edge_path_tdm_ratios = []
        for col_idx, net in enumerate(solver.nets):
            net.directed_tdm_edge_to_sink_idx_list = {}
            for offset, sol in enumerate(net.routing_solutions):
                sol_tdm_ratio = np.zeros(2 * solver.n_tdm_edge)
                for u, v in zip(sol[0:], sol[1:]):
                    if solver.weighted_routing_resource_net[u][v]['type'] == 1:
                        e = (u, v)
                        net.directed_tdm_edge_to_sink_idx_list.setdefault(e, []).append(offset)

                        r = solver.tdm_edge_row_idx_map[(min(u, v), max(u, v))]
                        sol_tdm_ratio[directed_tdm_edge_row_idx_map[e]] = solved_var_matrix[r, col_idx]

                dir_tdm_edge_path_tdm_ratios.append(sol_tdm_ratio)

        """
        Numpy.Array: (2 * n_tdm_edges, n_paths) 
        """
        dir_tdm_edge_path_tdm_ratio_matrix = np.vstack(dir_tdm_edge_path_tdm_ratios).T

        # delay = np.max(np.sum(dir_tdm_edge_path_tdm_ratio_matrix, axis=0) + path_bias)
        # assert abs(delay - primal_obj_val) < 1e-5

        save_experiment_results([solver.nets,
                                 solver.weighted_routing_resource_net,
                                 dir_tdm_edge_path_tdm_ratio_matrix,
                                 directed_tdm_edge_row_idx_map],
                                kwargs, res_dir, 'conti_tdm_ratio.pkl')












