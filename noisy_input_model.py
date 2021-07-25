from collections import defaultdict
from typing import List

import brian2 as bs
import numpy as np
import os
from matplotlib import pyplot as plt

from tqdm import tqdm

MEMBRANE_TIME_CONSTANT = 10 * bs.ms
V_REST = 0


def exp_escape_function(v, v_th=1):
    FIRE = True
    tau = 1.3
    beta = 1
    p = 1 / tau * np.exp(beta * (v - v_th))
    if np.random.uniform(low=0.0, high=1.0, size=1) < p:
        return FIRE
    else:
        return not FIRE


def mean_and_std(arr: List):
    arr = np.array(arr)

    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)

    return mean, std


def variance(arr):
    return


def noisy_input_test_vary_exc_inh(sim_time_in_ms=100):
    v_r = V_REST
    tau_m = MEMBRANE_TIME_CONSTANT
    lif_stein_model = """
                    dv/dt = (-(v-v_r)+ I)/tau_m : 1 (unless refractory)
                    I : 1 
                    """

    tau_t = 2.2
    v_th = 5
    beta = 10
    threshold = "v > v_th"

    reset = 'v=0'
    refractory = 1 * bs.ms

    # rates_magnitudes = [1, 5, 10, 20, 50, 100, 500, 1000]
    # bias_currents = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]

    # rates_magnitudes = [1, 5, 10]
    rates_magnitudes = [10, 50, 100]
    exc_weights = [1, 2, 5, 8]
    w = 0.1
    for exc_weight in exc_weights:
        for rate_magnitude in tqdm(rates_magnitudes):

            n_trials = 100
            v_trajectories = []
            plt.clf()

            for trial in range(n_trials):
                bs.start_scope()
                G = bs.NeuronGroup(N=1, model=lif_stein_model, threshold=threshold, reset=reset,
                                   refractory=refractory,
                                   method='euler')
                # give every neurons same start and same input to compare
                G.v = 0.0
                G.I = 0

                E = bs.PoissonGroup(100, rates=rate_magnitude * bs.Hz)
                I = bs.PoissonGroup(100, rates=rate_magnitude * bs.Hz)

                excitatory_connection = bs.Synapses(E, G, on_pre=f'v_post += {exc_weight}*w')
                inhibitory_connection = bs.Synapses(I, G, on_pre='v_post -= w')

                excitatory_connection.connect()
                inhibitory_connection.connect()

                state_monitor = bs.StateMonitor(G, 'v', record=True)
                spike_monitor = bs.SpikeMonitor(G)

                bs.run(sim_time_in_ms * bs.ms)

                v_trajectories.append(state_monitor.v[0])

            mean_trajectory = np.vstack(v_trajectories)
            mean_trajectory = np.mean(mean_trajectory, axis=0).squeeze()

            for v_trajectory in v_trajectories:
                plt.plot(state_monitor.t / bs.ms, v_trajectory, 'b', alpha=0.3)

            plt.plot(state_monitor.t / bs.ms, mean_trajectory, 'r', linestyle=":")
            bs.start_scope()
            G = bs.NeuronGroup(N=1, model=lif_stein_model, threshold=threshold, reset=reset,
                               refractory=refractory,
                               method='euler')
            # give every neurons same start and same input to compare

            G.v = 0.0
            G.I = 1.0
            state_monitor = bs.StateMonitor(G, 'v', record=True)

            bs.run(sim_time_in_ms * bs.ms)

            plt.plot(state_monitor.t / bs.ms, state_monitor.v[0], 'r')
            plt.xlabel('Time (ms)')
            plt.ylabel('u')
            plt.title(f"Rate: {rate_magnitude}, Exc_Inh_ratio: {exc_weight}, Bias Current: {1.0}")

            dir_name = f"vary_exc_inh_ratio"
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            plt.savefig(f"{dir_name}/Rate_{rate_magnitude}_ExcInhRatio_{exc_weight}_Bias_Current_{1.0}.png")
            plt.show()


def noisy_input_const_exc_inh(sim_time_in_ms=100):
    v_r = V_REST
    tau_m = MEMBRANE_TIME_CONSTANT
    lif_stein_model = """
                    dv/dt = (-(v-v_r)+ I)/tau_m : 1 (unless refractory)
                    I : 1 
                    """

    tau_t = 2.2
    v_th = 5
    beta = 10
    threshold = "v > v_th"

    reset = 'v=0'
    refractory = 1 * bs.ms
    w = 0.1

    # rates_magnitudes = [1, 5, 10, 20, 50, 100, 500, 1000]
    # bias_currents = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]

    # rates_magnitudes = [1, 5, 10]
    rates_magnitudes = [10, 50, 100]
    bias_currents = [1.0, 1.5, 2.0]
    for bias_current in bias_currents:
        for rate_magnitude in tqdm(rates_magnitudes):

            n_trials = 100
            v_trajectories = []
            plt.clf()

            for trial in range(n_trials):
                bs.start_scope()
                G = bs.NeuronGroup(N=1, model=lif_stein_model, threshold=threshold, reset=reset,
                                   refractory=refractory,
                                   method='euler')
                # give every neurons same start and same input to compare
                G.v = 0.0
                G.I = bias_current

                E = bs.PoissonGroup(100, rates=rate_magnitude * bs.Hz)
                I = bs.PoissonGroup(100, rates=rate_magnitude * bs.Hz)

                excitatory_connection = bs.Synapses(E, G, on_pre='v_post += w')
                inhibitory_connection = bs.Synapses(I, G, on_pre='v_post -= w')

                excitatory_connection.connect()
                inhibitory_connection.connect()

                state_monitor = bs.StateMonitor(G, 'v', record=True)
                spike_monitor = bs.SpikeMonitor(G)

                bs.run(sim_time_in_ms * bs.ms)

                v_trajectories.append(state_monitor.v[0])

            stacked_trajs = np.vstack(v_trajectories)
            mean_trajectory = np.mean(stacked_trajs, axis=0).squeeze()

            trajectory_stdev = np.std(stacked_trajs, axis=0).squeeze()
            traj = None
            for v_trajectory in v_trajectories:
                traj, = plt.plot(state_monitor.t / bs.ms, v_trajectory, 'b', alpha=0.15)

            fokker_planck_stdev = np.zeros_like(mean_trajectory)
            for _ in range(mean_trajectory):
                fokker_planck_stdev = tau_m * 1

            # mean_traj, = plt.plot(state_monitor.t / bs.ms, mean_trajectory, 'r', linestyle=":", label = "Mean Trajectory")
            error = plt.errorbar(state_monitor.t / bs.ms, mean_trajectory, trajectory_stdev, ecolor="orangered", linestyle=":", color="red")

            bs.start_scope()
            G = bs.NeuronGroup(N=1, model=lif_stein_model, threshold=threshold, reset=reset,
                               refractory=refractory,
                               method='euler')
            # give every neurons same start and same input to compare

            G.v = 0.0
            G.I = bias_current
            state_monitor = bs.StateMonitor(G, 'v', record=True)

            bs.run(sim_time_in_ms * bs.ms)

            det_traj, = plt.plot(state_monitor.t / bs.ms, state_monitor.v[0], 'k', label="Deterministic")
            plt.xlabel('Time (ms)')
            plt.ylabel('u')
            plt.title(f"Rate: {rate_magnitude}, Bias Current: {bias_current}")

            dir_name = f"const_exc_inh_weight"
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            plt.legend([traj, error, det_traj], ["Single trajectory", "Mean and Std", "Bias current trajectory"])
            plt.savefig(f"{dir_name}/Rate_{rate_magnitude}_Bias_Current: {bias_current}.png")
            plt.show()


if __name__ == '__main__':
    noisy_input_const_exc_inh(sim_time_in_ms=50)
    # noisy_input_test_vary_exc_inh(sim_time_in_ms=50)
