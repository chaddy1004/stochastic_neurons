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

    threshold = "rand()  < ( (1.0 / tau_t) * e**(beta * (v - v_th)))"
    # threshold = "v > v_th"

    reset = 'v=0'
    refractory = 1 * bs.ms
    G = bs.NeuronGroup(N=n_neurons, model=lif, threshold=threshold, reset=reset, refractory=refractory, method='euler')
    state_monitor = bs.StateMonitor(G, 'v', record=True)
    spike_monitor = bs.SpikeMonitor(G)

    # currents = [0.0, 0.01, 0.1, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0, 100, 300, 500, 1000]
    currents = [0.7, 1.0, 1.5, 2.0, 5.0]
    # currents = [100]
    bs.store()
    for current in currents:
        bs.restore()
        G.v = 0.0
        G.I = current
        bs.run(50 * bs.ms)

        # plt.clf()
        # plt.plot(state_monitor.t / bs.ms, state_monitor.v[0])
        # plt.xlabel('Time (ms)')
        # plt.ylabel('u (V)')
        # plt.title(f"Input Current of {current}A")
        # plt.show()

        # spike
        plt.clf()
        plt.plot(spike_monitor.t / bs.ms, spike_monitor.i, 'pk')  # the .k -> p means point marker, k means black
        plt.xlabel('Time (ms)')
        plt.ylabel('spikes')
        plt.title(f"Input Current of {current}A")
        plt.show()

    # # spike
    # plt.plot(spike_monitor.t / bs.ms, spike_monitor.i, 'pk')  # the .k -> p means point marker, k means black
    # plt.xlabel('Time (ms)')
    # plt.ylabel('spikes')
    # plt.show()
    #
    # plt.clf()
    # # You can also plot the actual voltage change of each neuron if you recorded it with state monitors
    # plt.plot(state_monitor.t / bs.ms, state_monitor.v[0])
    # plt.xlabel('Time (ms)')
    # plt.ylabel('v')
    # plt.title(f"Input Current of {}A")
    # plt.show()


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
    # ISI list per neuron on the population
    neuron_isi_lists = defaultdict(list)
    # ISI list over entire neuron list. This is to get the overall isi mean and variance of the neuronal population for given driving current
    overall_isi_lists = []
    last_spikes = defaultdict(int)

    @bs.network_operation(when="end")
    def prob(t):
        nonlocal last_length
        # print(f"Time: {t}")
        # print((1.0 / tau_t) * np.e ** (beta * (G.v - v_th)))
        curr_length = spike_monitor.t.shape[0]

        input_curr = np.random.normal(current, 0.3 * current)
        G.I = input_curr

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
                overall_isi_lists.append(curr_isi)
                last_spikes[spiked_neuron_index] = spike_time

        # print(last_length, curr_length)
        last_length = curr_length  # update the starting index for next set of spikes

    reset = 'v=0'
    refractory = 1 * bs.ms
    G = bs.NeuronGroup(N=n_neurons, model=lif, threshold=threshold, reset=reset, refractory=refractory, method='euler')
    state_monitor = bs.StateMonitor(G, 'v', record=True)
    spike_monitor = bs.SpikeMonitor(G)

    # currents = [0.0, 0.01, 0.1, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0, 100, 300, 500, 1000]
    currents = [0.01, 0.1, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0, 50, 500]
    bs.store()
    total_fig = plt.figure()
    ax_total = total_fig.add_subplot(111, title="Mean and Std for all currents for each neuron")

    mean_per_current = []
    std_per_current = []

    dir_name = f"soft_threshold_plots_{n_neurons}_{sim_time}"

    for current in tqdm(currents):
        neuron_isi_lists = defaultdict(list)
        overall_isi_lists = []
        last_spikes = defaultdict(int)

        bs.restore()
        G.v = 0.0
        # G.I = current


        # print(input_curr)

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
        fig.tight_layout(pad=3.0)

        overall_isis_for_current = np.array(overall_isi_lists)
        mean_per_current.append(np.mean(overall_isis_for_current))
        std_per_current.append(np.std(overall_isis_for_current))

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

        # plot isi for each neuron
        neurons = []
        means = []
        stds = []
        for neuron_index, mean_std in isi_mean_var.items():
            neurons.append(neuron_index)
            means.append(mean_std[0])
            stds.append(mean_std[1])
        ax2.errorbar(neurons, means, stds, linestyle="None", marker='^')
        if current == 500:
            ax2.set_ylim([0.95, 1.05])


        ax2.set_xlabel('Neuron #')
        ax2.set_ylabel('ISI (ms)')
        # plt.show()

        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        fig.savefig(f"{dir_name}/{current}A_driving_current.png")
        #
        # if current == 500:
        #     print(means)
        #     print(stds)
        #     exit(0)

        ax_total.errorbar(neurons, means, stds, linestyle="None", marker='^', label=f"{current} A")
        ax_total.hlines(y=1, xmin=0, xmax=len(neurons), linestyles="dashdot")
        ax_total.set_yscale("log")
        ax_total.set_ylabel("ISI (ms)")
        ax_total.set_xlabel("Neuron #")

    ax_total.legend(loc="best", title="Driving Current")
    total_fig.savefig(f"{dir_name}/current_injection{n_neurons}neurons_{sim_time}ms_sim_time_log_scale.png")

    total_fig = plt.figure()
    ax_total = total_fig.add_subplot(111, title=f"Mean and Std of ISI per input current, {n_neurons} neurons")
    for i, current in enumerate(currents):
        ax_total.errorbar(current, mean_per_current[i], std_per_current[i], linestyle="None", marker='^',
                          label=f"{current}A driving current")
        ax_total.text(current, mean_per_current[i], f"{current}A")
    ax_total.hlines(y=1, xmin=currents[0], xmax=currents[-1], linestyles="dashdot")
    ax_total.set_ylabel("ISI (ms)")
    ax_total.set_xlabel("Driving Current (A)")
    ax_total.set_xscale("log")
    ax_total.set_yscale("log")
    total_fig.savefig(f"{dir_name}/test_{n_neurons}_log.png")


if __name__ == '__main__':
    # main()
    # multiple_neurons(n_neurons=5, sim_time=500)
    # multiple_neurons(n_neurons=10, sim_time=500)
    # multiple_neurons(n_neurons=100, sim_time=500)
    multiple_neurons(n_neurons=200, sim_time=500)
