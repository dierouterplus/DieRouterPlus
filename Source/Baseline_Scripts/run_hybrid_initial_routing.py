import json
import sys

sys.path.append('../../')

from Source.Baseline.baseline_utils.metrics import get_net_criticality
from pathlib import Path
import time
import os
import argparse
from Source.utils.parse_result import check_result_exists
from Source.Baseline.baseline_utils.process_raw_data import build_weighted_routing_resource_network_and_nets
from Source.Baseline.HybridInitialRouting.hybrid_initial_routing import HybridInitialRouting
from Source.utils.save_result import save_experiment_results

def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid Initial Routing Configuration")

    parser.add_argument(
        '--testcase_dir',
        type=str,
        help='Directory containing the test cases.'
    )

    parser.add_argument(
        '--res_dir',
        type=str,
        default='../../Res',
        help='Directory to store the results.'
    )

    parser.add_argument(
        '--critical_net',
        type=str,
        default='SMT',
        choices=['SMT', 'Dijkstra', 'MST'],
        help='Routing method for critical nets. Choices are "SMT" or "Dijkstra".'
    )

    parser.add_argument(
        '--non_critical_net',
        type=str,
        default='Dijkstra',
        choices=['SMT', 'Dijkstra', 'MST'],
        help='Routing method for non-critical nets. Choices are "SMT" or "Dijkstra".'
    )

    parser.add_argument(
        '--n_pins_gap_factor',
        type=int,
        default=60,
        help='Gap factor to determine critical nets and non-critical nets.'
    )

    parser.add_argument(
        '--token',
        type=str,
        default='Hybrid-Initial-Routing',
        help='Token name for the routing process.'
    )

    return parser.parse_args()

if __name__ == "__main__":

    """
    'testcase_dir': f'../Data/testcase10',
    'res_dir': f'../Res/Baseline',
    'routing_config': {'critical_net':'SMT', 'non_critical_net': 'Dijkstra'},
    # 'routing_config': {'critical_net': 'Dijkstra', 'non_critical_net': 'SMT'},
    # 'routing_config': {'critical_net': 'Dijkstra', 'non_critical_net': 'Dijkstra'},
    'n_pins_gap_factor': 60,
    'token': 'Hybrid-Initial-Routing-SMT-Dijkstra'
    """

    config = parse_args()
    print("Configuration:")
    print(json.dumps(vars(config), indent=4))

    routing_config = {
        'critical_net': config.critical_net,
        'non_critical_net': config.non_critical_net
    }

    kwargs = {
        'testcase_dir': config.testcase_dir,
        'res_dir': config.res_dir,
        'routing_config': routing_config,
        'n_pins_gap_factor': config.n_pins_gap_factor,
        'token': config.token
    }


    kwargs['token'] = '-'.join([kwargs['token'],
                                kwargs['routing_config']['critical_net'],
                                kwargs['routing_config']['non_critical_net']])


    testcase_dir = Path(kwargs['testcase_dir'])
    testcase = testcase_dir.name
    res_dir = Path(kwargs['res_dir']) / testcase

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)


    is_result_present, _ = check_result_exists(res_dir, kwargs)

    if is_result_present:
        print(f"Result already presents in {_}")
    else:
        weighted_routing_resource_network, nets = build_weighted_routing_resource_network_and_nets(testcase_dir)
        hybrid_initial_routing = HybridInitialRouting(nets,
                                                      weighted_routing_resource_network,
                                                      routing_config=kwargs['routing_config'],
                                                      n_pins_gap_factor=kwargs['n_pins_gap_factor'])

        start_time = time.time()
        nets, weighted_routing_resource_network = hybrid_initial_routing.perform_routing()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Hybrid Initial Routing Finished in:{elapsed_time}s")

        """
        The result includes two parts: nets and weighted_routing_resource_network.
        """
        save_experiment_results([nets, weighted_routing_resource_network],
                                kwargs, res_dir, 'hyb_init_routing_res.pkl')

        system_delay = max([get_net_criticality(weighted_routing_resource_network, net) for net in nets])
        print('System delay after Hybrid Initial Routing is {}'.format(system_delay))










