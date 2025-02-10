import numpy as np
from Baseline.Discretization.dp_discretization_mixin import DPDiscretizationMixin
class IncrementalDPDiscretization(DPDiscretizationMixin):
    def __init__(self, net_to_conti_tdm_ratio, resource_count):
        """
        :param net_to_conti_tdm_ratio: dict(net -> continuous tdm ratio)
        :param resource_count: the total number of resources
        """
        self.net_conti_tdm_ratio_pairs = list(net_to_conti_tdm_ratio.items())
        self.resource_count = resource_count

        """
        r_0 <= r_1 <= ... <= r_{m-1}
        """
        self.net_conti_tdm_ratio_pairs.sort(key=lambda x: x[1])
        self.n_net = len(self.net_conti_tdm_ratio_pairs)

        """
        self.cost: a cost vector under a certain number of resources. 
        self.cost[i]: the cost of allocating net 0 ~ net i to a certain number of resources.
        ass[i, j]: the last ass[i, j] nets (net i - ass[i][j] + 1 ~ net i) are assigned to 
        the j-th physical wire.
        
        When n_net > 0, self.cost is a (n_net, ) array; 
        self.ass is a (n_net, resource_count) array. 
        
        when n_net = 0, self.cost is a (1, ) array and its value is -1 * inf.
        self.ass is None
        """
        if self.n_net == 0:
            self.cost = np.full((1, ), -1 * np.inf)
            self.n_allocated_resources = 0
            self.ass = None
        else:
            """
            cost[i]: the objective value obtained by assigning net 0 ~ net i to a 
            certain number of resources.
            The initial number of resource is 1.
            """
            self.cost = np.array([self.r(k) - self.net_conti_tdm_ratio_pairs[0][1] for k in range(1, self.n_net + 1)])
            self.n_allocated_resources = 1
            self.ass = np.zeros((self.n_net, self.resource_count), dtype=int)
            self.ass[:, 0] = np.array([k for k in range(1, self.n_net + 1)])
    def increase_resource_and_update_cost(self):
        """
        Add a physical wire and update the cost vector.
        :return:
        """

        """
        (n_nets,)
        """
        updated_cost = self.cost.copy()
        self.n_allocated_resources += 1

        s = 1

        """
        State i : Nets to be assigned are net 0 ~ net i.
        
        When the number of nets is fewer than the number of allocated resources, 
        the previously computed results can be used directly.
        """
        for i in range(self.n_allocated_resources - 1, self.n_net):
            """
            Action s : Allocate net 0 ~ net s-1 to previous wires, 
            allocate net s ~ net n_net - 1 to the newly added wire.
            
            Q(i, s)
            """
            q_i_s = max([self.cost[s - 1], self.r(i + 1 - s) - self.net_conti_tdm_ratio_pairs[s][1]])

            while s < i:
                """
                q_i_s < q_i_s_plus_1 <==> s is the optimal action
                If s == i, then s must be the optimal action
                """
                q_i_s_plus_1 = max([self.cost[s], self.r(i - s) - self.net_conti_tdm_ratio_pairs[s + 1][1]])
                if q_i_s < q_i_s_plus_1:
                    break
                q_i_s = q_i_s_plus_1
                s += 1

            """
            q_i_s must be the optimal value
            """
            updated_cost[i] = q_i_s
            self.ass[i, self.n_allocated_resources-1] = i + 1 - s

        self.cost = updated_cost



if __name__ == '__main__':
    n_net = 50000
    resource_cnt = 500
    import random
    random.seed(0)
    tdm_ratios = [random.uniform(0, 10) for _ in range(n_net)]
    net_to_tdm_ratio = dict(zip(range(n_net), tdm_ratios))






