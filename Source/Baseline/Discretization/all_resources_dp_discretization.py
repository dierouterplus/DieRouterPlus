from Source.Baseline.Discretization.dp_discretization_mixin import DPDiscretizationMixin

import numpy as np
from Source.utils.timing import timeit
class AllResourcesDPDiscretization(DPDiscretizationMixin):
    def __init__(self, net_to_conti_tdm_ratio, resource_count):
        """
        :param net_to_conti_tdm_ratio: dict(net -> continuous tdm ratio)
        :param resource_count: number of available tdm type wires
        """
        self.net_conti_tdm_ratio_pairs = list(net_to_conti_tdm_ratio.items())
        self.resource_count = resource_count

        """
        r_0 <= r_1 <= ... <= r_{m-1}
        """
        self.net_conti_tdm_ratio_pairs.sort(key=lambda x: x[1])
        self.n_net = len(self.net_conti_tdm_ratio_pairs)

        """
        dp[i][j]: the objective value of assigning
        net 0 ~ net i to wire 0 ~ wire j.
        """
        self.dp = np.zeros((self.n_net, self.resource_count))
        self.dp[0] = self.r(1) - self.net_conti_tdm_ratio_pairs[0][1]
        self.dp[:, 0] = np.array([self.r(k) - self.net_conti_tdm_ratio_pairs[0][1] for k in range(1, self.n_net + 1)])

        """
        ass[i][j]: In state (i, j), the last ass[i][j] nets (nets i - ass[i][j] + 1 to i) are assigned to the j-th physical wire.
        """
        self.ass = np.zeros((self.n_net, self.resource_count), dtype=int)
        self.ass[0, 0] = 1
        self.ass[:, 0] = np.array([k for k in range(1, self.n_net + 1)])

    @timeit("perform_dp")
    def perform_dp(self):
        """
        :return: cost vector according to resource count 1 ~ self.resource_count.
        """
        for j in range(1, self.resource_count):
            """
            Action s : Allocate net 0 ~ net s-1 to wire 0 ~ wire j-1 and net s ~ net i to wire j.
            1 <= s <= i
            """
            s = 1

            for i in range(j, self.n_net):
                """
                Value of applying action s under state (i, j) 
                """
                q_s = max([self.dp[s - 1, j - 1], self.r(i + 1 - s) - self.net_conti_tdm_ratio_pairs[s][1]])

                while s < i:
                    """
                    q_s < q_s_plus_one <==> s is the optimal action
                    If s == i, then s must be the optimal action
                    """
                    q_s_plus_one = max([self.dp[s,j-1],  self.r(i-s) - self.net_conti_tdm_ratio_pairs[s+1][1]])
                    if q_s < q_s_plus_one:
                        break
                    q_s = q_s_plus_one
                    s += 1

                """
                q_s must be the optimal value
                """
                self.dp[i, j] = q_s
                self.ass[i, j] = i + 1 - s

                """
                Deal with scenarios with abundant resources.
                """
                if i == j:
                    self.dp[i, j+1: ] = self.dp[i, j]
                    self.ass[i, j+1: ] = 0

        return self.dp[self.n_net - 1]








if __name__ == '__main__':
    n_net = 2
    resource_cnt = 2
    import random
    random.seed(0)
    tdm_ratios = [random.uniform(0, 10) for _ in range(n_net)]
    net_to_tdm_ratio = dict(zip(range(n_net), tdm_ratios))

    dp = AllResourcesDPDiscretization(net_to_tdm_ratio, resource_cnt)
    res = dp.perform_dp()
    print(np.mean(res), np.min(res), np.max(res))





