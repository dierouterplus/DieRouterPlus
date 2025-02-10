import pickle
import sys

sys.path.append('../../')
from Source.Baseline.PerfDrivenRipReroute.perf_driven_rip_reroute_net import PerfDrivenRipRerouteNet
from Source.Baseline.VioRipReroute.vio_rip_reroute import VioRipReroute

import time
from Source.utils.parse_result import check_result_exists, get_previous_step_output_path
from Source.utils.save_result import save_experiment_results

import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Routing Configuration")

    # 基本参数
    parser.add_argument(
        '--up_dir',
        type=str,
        help='Directory for previous results.'
    )

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

    # token 参数
    parser.add_argument(
        '--token',
        type=str,
        default='Two-Stage-Reroute',
        help='Token name for the routing process.'
    )

    # 解析参数
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
            }
        },
        'edge_criticality_metric': 'Max',
        # 'edge_criticality_metric': 'Sum',
        'token': 'Two-Stage-Reroute'
    }
    """

    config = parse_args()
    print("Configuration:")
    print(json.dumps(vars(config), indent=4))

    # construct identifier
    identifier = {
        '0': {
            'n_pins_gap_factor': config.identifier_0_n_pins_gap_factor,
            'token': config.identifier_0_token
        }
    }

    # 构建最终的 kwargs 字典
    kwargs = {
        'up_dir': config.up_dir,
        'identifier': identifier,
        'edge_criticality_metric': config.edge_criticality_metric,
        'token': config.token
    }

    kwargs['token'] = '-'.join([kwargs['token'], kwargs['edge_criticality_metric']])

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
        perf_driven_reroute = PerfDrivenRipRerouteNet(vio_rip_rerouter.nets,
                                                      vio_rip_rerouter.weighted_routing_resource_net,
                                                      vio_rip_rerouter.edge_to_routing_net_set)

        opt_nets, opt_weighted_routing_resource_net, opt_system_delay = perf_driven_reroute.perf_driven_rip_reroute()
        end = time.time()
        print('Performance-Driven Rip Up and Reroute took {} seconds'.format(end - start))
        print('Opt system delay is {}'.format(opt_system_delay))

        save_experiment_results([opt_nets, opt_weighted_routing_resource_net, perf_driven_reroute.edge_to_routing_net_set],
                                kwargs, res_dir, 'two_stage_reroute_res.pkl')












