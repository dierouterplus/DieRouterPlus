import json
import pickle
import sys

sys.path.append('../')
from Baseline.PerfDrivenRipReroute.perf_driven_rip_reroute_sink import PerfDrivenRipRerouteSink
from Baseline.VioRipReroute.vio_rip_reroute import VioRipReroute

import time
from utils.parse_result import check_result_exists, get_previous_step_output_path
from utils.save_result import save_experiment_results

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Two Stage Rerouting Configuration")

    # up_dir 参数
    parser.add_argument(
        '--up_dir',
        type=str,
        help='Directory for previous results.'
    )

    # Identifier 0 参数
    parser.add_argument(
        '--identifier_0_n_pins_gap_factor',
        type=int,
        # default=60,
        help='Gap factor for the number of pins in identifier 0.'
    )

    parser.add_argument(
        '--identifier_0_token',
        type=str,
        # default='Hybrid-Initial-Routing-SMT-Dijkstra',
        help='Token for identifier 0.'
    )

    parser.add_argument(
        '--edge_criticality_metric',
        type=str,
        default='Max',
        choices=['Max', 'Sum'],
        help='Metric for edge criticality. Choices are "Max" or "Sum".'
    )

    parser.add_argument(
        '--occupied_edge_cost',
        type=str,
        choices=['zero', 'weight'],
        default='zero',
        help='Cost metric for occupied edges. Choices are "zero" or "weight".'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help=''
    )



    parser.add_argument(
        '--token',
        type=str,
        default='Two-Stage-Reroute-Sink',
        help='Token name for the routing process.'
    )

    args = parser.parse_args()
    return args





if __name__ == "__main__":
    """
    {
        'up_dir': '../Res/Baseline/testcase10',
        'identifier': {
            '0': {
                'n_pins_gap_factor': 60,
                'token': 'Hybrid-Initial-Routing-SMT-Dijkstra'
            }
        },
        'edge_criticality_metric': 'Max',
        # 'edge_criticality_metric': 'Sum',
        'occupied_edge_cost': 'zero',
        # 'occupied_edge_cost': 'weight',
        'token': 'Two-Stage-Reroute-Sink'
    }
    """
    args = parse_args()
    print("Configuration:")
    print(json.dumps(vars(args), indent=4))

    identifier = {
        '0': {
            'n_pins_gap_factor': args.identifier_0_n_pins_gap_factor,
            'token': args.identifier_0_token
        }
    }

    kwargs = {
        'up_dir': args.up_dir,
        'identifier': identifier,
        'edge_criticality_metric': args.edge_criticality_metric,
        'occupied_edge_cost': args.occupied_edge_cost,
        'patience': args.patience,
        'token': args.token
    }

    kwargs['token'] = '-'.join([kwargs['token'],
                                kwargs['edge_criticality_metric'],
                                kwargs['occupied_edge_cost']])

    res_dir = get_previous_step_output_path(kwargs['up_dir'], kwargs['identifier'])
    is_result_present, _ = check_result_exists(res_dir, kwargs)

    if is_result_present:
        print(f"Result already presents in {_}")
    else:
        hybrid_init_routing_res_path = f"{res_dir}/hyb_init_routing_res.pkl"
        with open(hybrid_init_routing_res_path, "rb") as f:
            nets, weighted_routing_resource_network = pickle.load(f)

        start = time.time()
        vio_rip_rerouter = VioRipReroute(nets, weighted_routing_resource_network)
        vio_rip_rerouter.rip_up_and_reroute()
        end = time.time()
        print('Violation Rip Up and Reroute took {} seconds'.format(end - start))

        start = time.time()
        perf_driven_reroute = PerfDrivenRipRerouteSink(vio_rip_rerouter.nets,
                                                      vio_rip_rerouter.weighted_routing_resource_net,
                                                      vio_rip_rerouter.edge_to_routing_net_set,
                                                       kwargs['edge_criticality_metric'],
                                                       kwargs['occupied_edge_cost'],
                                                       kwargs['patience'])

        opt_nets, opt_weighted_routing_resource_net, opt_system_delay = perf_driven_reroute.perf_driven_rip_reroute()
        end = time.time()
        print('Performance-Driven Rip Up and Reroute took {} seconds'.format(end - start))
        print('Opt system delay is {}'.format(opt_system_delay))

        save_experiment_results([opt_nets, opt_weighted_routing_resource_net, perf_driven_reroute.edge_to_routing_net_set],
                                kwargs, res_dir, 'two_stage_reroute_res.pkl')












