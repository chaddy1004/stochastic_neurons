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


def main():
    v_r = V_REST
    tau_m = MEMBRANE_TIME_CONSTANT
    lif = """
    dv/dt = (-(v-v_r) + I)/tau_m: 1 (unless refractory)
    I : 1 
    """
    n_neurons = 1
    bs.start_scope()

    tau_t = 2.2
    v_th = 1.0
    beta = 10

    threshold = "rand() < ( (1.0 / tau_t) * e**(beta * (v - v_th)))"

    reset = 'v=0'
    refractory = 1 * bs.ms
    G = bs.NeuronGroup(N=n_neurons, model=lif, threshold=threshold, reset=reset, refractory=refractory, method='euler')
    state_monitor = bs.StateMonitor(G, 'v', record=True)
    spike_monitor = bs.SpikeMonitor(G)

    currents = [0.0, 0.01, 0.1, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0, 100, 300, 500, 1000]
    bs.store()
    for current in currents:
        bs.restore()
        G.v = 0.0
        G.I = current
        bs.run(100 * bs.ms)

    # spike
    plt.plot(spike_monitor.t / bs.ms, spike_monitor.i, 'pk')  # the .k -> p means point marker, k means black
    plt.xlabel('Time (ms)')
    plt.ylabel('spikes')
    plt.show()

    plt.clf()
    # You can also plot the actual voltage change of each neuron if you recorded it with state monitors
    plt.plot(state_monitor.t / bs.ms, state_monitor.v[0])
    plt.xlabel('Time (ms)')
    plt.ylabel('v')
    plt.show()


def multiple_neurons(n_neurons, sim_time):
    v_r = V_REST
    tau_m = MEMBRANE_TIME_CONSTANT
    lif = """
    dv/dt = (-(v-v_r) + I)/tau_m: 1 (unless refractory)
    I : 1 
    """
    bs.start_scope()

    tau_t = 2.2
    v_th = 1.0
    beta = 10
    threshold = "rand() < ( (1.0 / tau_t) * e**(beta * (v - v_th)))"
    # threshold = "v > v_th"

    reset = 'v=0'
    refractory = 1 * bs.ms
    G = bs.NeuronGroup(N=n_neurons, model=lif, threshold=threshold, reset=reset, refractory=refractory, method='euler')
    # give every neurons same start and same input to compare
    G.v = 0.0
    G.I = 1.5

    state_monitor = bs.StateMonitor(G, 'v', record=True)
    spike_monitor = bs.SpikeMonitor(G)

    last_length = 0
    neuron_isi_lists = defaultdict(list)
    last_spikes = defaultdict(int)

    reset = 'v=0'
    refractory = 1 * bs.ms
    G = bs.NeuronGroup(N=n_neurons, model=lif, threshold=threshold, reset=reset, refractory=refractory, method='euler')
    state_monitor = bs.StateMonitor(G, 'v', record=True)
    spike_monitor = bs.SpikeMonitor(G)

    @bs.network_operation(when="end")
    def prob(t):
        nonlocal last_length
        # print(f"Time: {t}")
        # print((1.0 / tau_t) * np.e ** (beta * (G.v - v_th)))
        curr_length = spike_monitor.t.shape[0]

        if curr_length > last_length:
            spiked_neuron_indices = spike_monitor.i
            spike_times = spike_monitor.t
            for i in range(last_length, curr_length):
                spiked_neuron_index = spiked_neuron_indices[i]
                spike_time = spike_times[i] / bs.ms

                last_spike_time = last_spikes[spiked_neuron_index]

                # print("spike time, last_spike time", spike_time, last_spike_time)
                curr_isi = (spike_time - last_spike_time)  # /bs.ms
                neuron_isi_lists[spiked_neuron_index].append(curr_isi)
                last_spikes[spiked_neuron_index] = spike_time
                # print(spike_monitor.i[i], type(spike_monitor.i[i]))

        # print(last_length, curr_length)
        last_length = curr_length  # update the starting index for next set of spikes

    currents = [0.0, 0.01, 0.1, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0, 100, 300, 500, 1000]
    # currents = [500]
    bs.store()
    total_fig = plt.figure()
    ax_total = total_fig.add_subplot(111, title="Mean and Std for all currents")
    for current in tqdm(currents):

        neuron_isi_lists = defaultdict(list)
        last_spikes = defaultdict(int)
        bs.restore()
        G.v = 0.0
        G.I = current
        bs.run(sim_time * bs.ms)

        isi_mean_var = defaultdict(tuple)
        # fig, (ax0, ax1) = plt.subplots(2, sharex=True)
        fig = plt.figure(figsize=(12, 10))
        fig.subplots_adjust(top=0.8)

        plot1_title = "Membrane Potential"
        if n_neurons > 10:
            plot1_title = "Membrane Potential of neuron 0"
        ax0 = fig.add_subplot(311, title=plot1_title)
        ax1 = fig.add_subplot(312, sharex=ax0, title=f"Spikes caused by input with I={current}")
        ax2 = fig.add_subplot(313, title=f"Mean and StdDev of ISI for each neuron")

        for i in sorted(neuron_isi_lists.keys()):
            # print(i)
            # Not considering ISI of very first spike and zero since there are some edge cases that happens
            # specifically, when the driving current is too large, the first spike occurs at 0
            # but the first "last_spike_time" is also zero
            # so for those cases, I get an ISI of 0. I just wanted to avoid that.
            # It shouldnt matter in the long run, but i didnt want that to mess up the total mean and variance
            neuron_isi_lists[i] = neuron_isi_lists[i][1:]

            mean, std = mean_and_std(neuron_isi_lists[i])
            isi_mean_var[i] = (mean, std)

            # print(isi_mean_var)

            # just draw the spike plot once and draw the voltage
            # if i == 0:
            #     plt.clf()
            #     # You can also plot the actual voltage change of each neuron if you recorded it with state monitors
            if n_neurons <= 10:
                ax0.plot(state_monitor.t / bs.ms, state_monitor.v[i], label=f"{i}")
            elif n_neurons > 10:
                if i == 0:
                    ax0.plot(state_monitor.t / bs.ms, state_monitor.v[i], label=f"{i}")
            else:
                raise ValueError("Error")

        ax0.set_xlabel('Time (ms)')
        ax0.set_ylabel('v')
        if n_neurons <= 10:
            ax0.legend(loc="upper right", bbox_to_anchor=(1.1, 1), ncol=2)

        ax1.plot(spike_monitor.t / bs.ms, spike_monitor.i, 'pk')  # the .k -> p means point marker, k means black
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Neuron #')

        neurons = []
        means = []
        variances = []
        for neuron_index, mean_std in isi_mean_var.items():
            neurons.append(neuron_index)
            means.append(mean_std[0])
            variances.append(mean_std[1])
        ax2.errorbar(neurons, means, variances, linestyle="None", marker='^')
        ax2.set_xlabel('Neuron #')
        ax2.set_ylabel('ISI (ms)')
        # plt.show()
        dir_name = f"plots_{n_neurons}_{sim_time}"
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        fig.savefig(f"{dir_name}/{current}A_driving_current.png")

        ax_total.errorbar(neurons, means, variances, linestyle="None", marker='^', label=f"{current}")

    ax_total.legend(loc="best")
    total_fig.savefig(f"{n_neurons}neurons_{sim_time}ms_sim_time.png")
    total_fig.show()


if __name__ == '__main__':
    # main()
    multiple_neurons(n_neurons=5, sim_time=200)
    multiple_neurons(n_neurons=10, sim_time=200)
    multiple_neurons(n_neurons=100, sim_time=200)
    multiple_neurons(n_neurons=200, sim_time=200)
