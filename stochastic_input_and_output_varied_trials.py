from collections import defaultdict
from typing import List

import brian2 as bs
import numpy as np
import os
from matplotlib import pyplot as plt

from tqdm import tqdm

MEMBRANE_TIME_CONSTANT = 10 * bs.ms
V_REST = 0


def mean_and_std(arr: List):
    arr = np.array(arr)

    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)

    return mean, std


def visualise_connectivity(S, name):
    plt.clf()
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(bs.zeros(Ns), bs.arange(Ns), 'ok', ms=10)
    plt.plot(bs.ones(Nt), bs.arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plt.plot([0, 1], [i, j], '-k')
    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(Ns, Nt))
    plt.subplot(122)
    plt.plot(S.i, S.j, 'ok')
    plt.xlim(-1, Ns)
    plt.ylim(-1, Nt)
    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')
    # plt.show()
    plt.savefig(f"{name}.png")


def multiple_neurons_with_poisson_neurons(n_neurons, sim_time):
    v_r = V_REST
    tau_m = MEMBRANE_TIME_CONSTANT
    lif = """
    dv/dt = (-(v-v_r))/tau_m: 1 (unless refractory)
    I : 1 
    """
    bs.start_scope()

    tau_t = 2.2
    v_th = 1.0
    beta = 10
    threshold = "rand() < ( (1.0 / tau_t) * e**(beta * (v - v_th)))"

    reset = 'v=0'
    refractory = 1 * bs.ms

    rates_magnitude = [1, 5, 10, 20, 50, 100, 500, 1000]

    w = 0.1

    mean_per_rate = []
    std_per_rate = []

    all_neuron_fig = plt.figure()
    all_neuron_plot = all_neuron_fig.add_subplot(111,
                                                 title="Mean and Std for all rates for each neuron (Balanced Input)")
    for rate_magnitude in tqdm(rates_magnitude):
        # give every neurons same start and same input to compare

        last_length = 0
        neuron_isi_lists = defaultdict(list)
        overall_isi_lists = []
        last_spikes = defaultdict(int)

        reset = 'v=0'
        refractory = 1 * bs.ms
        G = bs.NeuronGroup(N=n_neurons, model=lif, threshold=threshold, reset=reset, refractory=refractory,
                           method='euler')
        state_monitor = bs.StateMonitor(G, 'v', record=True)
        spike_monitor = bs.SpikeMonitor(G)

        G.v = 0.0

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
                    overall_isi_lists.append(curr_isi)
                    last_spikes[spiked_neuron_index] = spike_time
                    # print(spike_monitor.i[i], type(spike_monitor.i[i]))

            # print(last_length, curr_length)
            last_length = curr_length  # update the starting index for next set of spikes

        E = bs.PoissonGroup(100, rates=rate_magnitude * bs.Hz)
        I = bs.PoissonGroup(100, rates=rate_magnitude * bs.Hz)

        excitatory_connection = bs.Synapses(E, G, on_pre='v_post += w')
        inhibitory_connection = bs.Synapses(I, G, on_pre='v_post -= w')

        excitatory_connection.connect()
        inhibitory_connection.connect()
        # visualise_connectivity(excitatory_connection, name="excit")
        # visualise_connectivity(inhibitory_connection, name="inhib")
        # exit(0)

        neuron_isi_lists = defaultdict(list)
        last_spikes = defaultdict(int)

        G.v = 0.0
        bs.run(sim_time * bs.ms)

        isi_mean_var = defaultdict(tuple)
        # fig, (ax0, ax1) = plt.subplots(2, sharex=True)
        fig = plt.figure(figsize=(12, 10))
        fig.subplots_adjust(top=0.8)

        plot1_title = "Membrane Potential"
        if n_neurons > 10:
            plot1_title = "Membrane Potential of neuron 0"
        ax0 = fig.add_subplot(311, title=plot1_title)
        ax1 = fig.add_subplot(312, sharex=ax0, title=f"Spikes caused by input with rate={rate_magnitude}")
        ax2 = fig.add_subplot(313, title=f"Mean and StdDev of ISI for each neuron")
        fig.tight_layout(pad=3.0)

        overall_isis_for_current = np.array(overall_isi_lists)
        mean_per_rate.append(np.mean(overall_isis_for_current))
        std_per_rate.append(np.std(overall_isis_for_current))

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
        dir_name = f"rate_plots_{n_neurons}_{sim_time}"
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        fig.savefig(f"{dir_name}/{rate_magnitude}Hz_presynaptic_spikes.png")

        all_neuron_plot.errorbar(neurons, means, variances, linestyle="None", marker='^', label=f"{rate_magnitude}Hz")

    all_neuron_plot.legend(loc="best")
    all_neuron_plot.set_yscale("log")
    all_neuron_fig.savefig(f"poisson_neuron_{n_neurons}neurons_{sim_time}ms_sim_time.png")

    all_rate_fig = plt.figure()
    all_rate_plot = all_rate_fig.add_subplot(111,
                                             title=f"Mean and Std of ISI per input rates, {n_neurons} neurons (Balanced Input)")
    for i, rate in enumerate(rates_magnitude):
        all_rate_plot.errorbar(rate, mean_per_rate[i], std_per_rate[i], linestyle="None", marker='^',
                               label=f"{rate}Hz Presynaptic input rate")
        all_rate_plot.text(rate, mean_per_rate[i], f"{rate}Hz")
    all_rate_plot.hlines(y=1, xmin=rates_magnitude[0], xmax=rates_magnitude[-1], linestyles="dashdot")
    all_rate_plot.set_ylabel("ISI (ms)")
    all_rate_plot.set_xlabel("Presynaptic Poisson Rate")
    all_rate_plot.set_xscale("log")
    all_rate_plot.set_yscale("log")
    all_rate_fig.savefig(f"rates_test_{n_neurons}_log.png")
    # total_fig.show()


# def multiple_neurons_with_poisson_neurons_unbalanced(n_trials, sim_time, weight):
#     v_r = V_REST
#     tau_m = MEMBRANE_TIME_CONSTANT
#     lif = """
#     dv/dt = (-(v-v_r))/tau_m: 1 (unless refractory)
#     I : 1
#     """
#     bs.start_scope()
#
#     tau_t = 2.2
#     v_th = 1.0
#     beta = 10
#     threshold = "rand() < ( (1.0 / tau_t) * e**(beta * (v - v_th)))"
#
#     reset = 'v=0'
#     refractory = 1 * bs.ms
#
#     # rates_magnitude = [1, 5, 10, 20, 50, 100, 500, 1000]
#     rate_magnitudes = [100]  # poisson process rate; hz
#
#     w = 0.001
#
#     isi_means_each_rates = []
#     isi_stds_each_rates = []
#
#     all_trial_fig = plt.figure()
#     all_trial_plot = all_trial_fig.add_subplot(111,
#                                                title=f"Mean and Std for all rates for each trials (Unbalanced Input, {weight})")
#
#     dir_name = f"{weight}_unbalanced_{n_trials}_trials_{sim_time}"
#     rates_isi_list = []
#     for rate_magnitude in rate_magnitudes:
#         trial_isi_list = []
#
#         fig = plt.figure(figsize=(12, 10))
#         fig.subplots_adjust(top=0.8)
#
#         plot1_title = "Membrane Potentials"
#
#         u_each_trial = fig.add_subplot(311, title=plot1_title)
#         spikes_each_trial = fig.add_subplot(312, sharex=u_each_trial,
#                                             title=f"Spikes caused by input with rate={rate_magnitude}")
#         isi_mean_std_each_trial = fig.add_subplot(313, title=f"Mean and StdDev of ISI for each neuron")
#         fig.tight_layout(pad=3.0)
#
#         isi_means_each_trials = []
#         isi_stds_each_trials = []
#
#         for trial in tqdm(range(n_trials)):
#             last_length = 0
#
#             last_spike_time = 0
#
#             reset = 'v=0'
#             refractory = 1 * bs.ms
#             G = bs.NeuronGroup(N=1, model=lif, threshold=threshold, reset=reset, refractory=refractory,
#                                method='euler')
#             state_monitor = bs.StateMonitor(G, 'v', record=True)
#             spike_monitor = bs.SpikeMonitor(G)
#
#             ############################################################################################################
#             ############################################################################################################
#             ############################################################################################################
#             @bs.network_operation(when="end")
#             def get_isi(t):
#                 nonlocal last_length
#                 nonlocal last_spike_time
#                 # print(f"Time: {t}")
#                 # print((1.0 / tau_t) * np.e ** (beta * (G.v - v_th)))
#                 curr_length = spike_monitor.t.shape[0]
#
#                 # if there was a new spike, the curr length of the spike monitor would be longer than the last
#                 if curr_length > last_length:
#                     spiked_neuron_indices = spike_monitor.i
#                     spike_times = spike_monitor.t
#                     for i in range(last_length, curr_length):
#                         spiked_neuron_index = spiked_neuron_indices[i]
#                         spike_time = spike_times[i] / bs.ms
#
#                         # print("spike time, last_spike time", spike_time, last_spike_time)
#                         curr_isi = (spike_time - last_spike_time)  # /bs.ms
#                         trial_isi_list.append(curr_isi)
#                         last_spike_time = spike_time
#                         # print(spike_monitor.i[i], type(spike_monitor.i[i]))
#
#                 # print(last_length, curr_length)
#                 last_length = curr_length  # update the starting index for next set of spikes
#
#             ############################################################################################################
#             ############################################################################################################
#             ############################################################################################################
#
#             E = bs.PoissonGroup(400, rates=rate_magnitude * bs.Hz)
#             I = bs.PoissonGroup(400, rates=rate_magnitude * bs.Hz)
#
#             excitatory_connection = bs.Synapses(E, G, on_pre=f'v_post += {weight}*w')
#             inhibitory_connection = bs.Synapses(I, G, on_pre='v_post -= w')
#
#             excitatory_connection.connect()
#             inhibitory_connection.connect()
#
#             G.v = 0.0
#             bs.run(sim_time * bs.ms)
#
#             isi_mean_var = defaultdict(tuple)
#             # fig, (ax0, ax1) = plt.subplots(2, sharex=True)
#
#             # Not considering ISI of very first spike and zero since there are some edge cases that happens
#             # specifically, when the driving current is too large, the first spike occurs at 0
#             # but the first "last_spike_time" is also zero
#             # so for those cases, I get an ISI of 0. I just wanted to avoid that.
#             # It shouldnt matter in the long run, but i didnt want that to mess up the total mean and variance
#             overall_isis_for_trial = np.array(trial_isi_list[1:])
#             isi_means_each_trials.append(np.mean(overall_isis_for_trial))
#             isi_stds_each_trials.append(np.std(overall_isis_for_trial))
#
#             u_each_trial.plot(state_monitor.t / bs.ms, state_monitor.v[0], label=f"{trial}")
#
#             u_each_trial.set_xlabel('Time (ms)')
#             u_each_trial.set_ylabel('v')
#             # if n_neurons <= 10:
#             #     ax0.legend(loc="upper right", bbox_to_anchor=(1.1, 1), ncol=2)
#
#             trial_index = np.array([trial for _ in range(spike_monitor.t.shape[0])])
#
#             # print(trial_index)
#
#             spikes_each_trial.plot(spike_monitor.t / bs.ms, trial_index, 'pk')  # pk -> p = point marker, k = black
#             spikes_each_trial.set_xlabel('Time (ms)')
#             spikes_each_trial.set_ylabel('Neuron #')
#
#             # neurons = []
#             # means = []
#             # variances = []
#             # for neuron_index, mean_std in isi_mean_var.items():
#             #     neurons.append(neuron_index)
#             #     means.append(mean_std[0])
#             #     variances.append(mean_std[1])
#             isi_mean_std_each_trial.errorbar(trial, isi_means_each_trials[-1], isi_stds_each_trials[-1], linestyle="None", marker='^')
#             isi_mean_std_each_trial.set_xlabel('Neuron #')
#             isi_mean_std_each_trial.set_ylabel('ISI (ms)')
#
#             # plt.show()
#             if not os.path.isdir(dir_name):
#                 os.mkdir(dir_name)
#             fig.savefig(f"{dir_name}/{rate_magnitude}Hz_presynaptic_spikes_{n_trials}_trials.png")
#
#         isi_means_each_rates.append(np.mean(np.array(isi_means_each_trials)))
#         isi_stds_each_rates.append(np.std(np.array(isi_stds_each_trials)))
#
#     all_rate_fig = plt.figure()
#     all_rate_plot = all_rate_fig.add_subplot(111,
#                                              title=f"Mean and Std of ISI per input rates, {n_trials} trials (Unbalanced Input, {weight})")
#     for i, rate in enumerate(rate_magnitudes):
#         all_rate_plot.errorbar(rate, isi_means_each_rates[i], isi_stds_each_rates[i], linestyle="None", marker='^',
#                                label=f"{rate}Hz Presynaptic input rate")
#         all_rate_plot.text(rate, isi_means_each_rates[i], f"{rate}Hz")
#     all_rate_plot.hlines(y=1, xmin=rate_magnitudes[0], xmax=rate_magnitudes[-1], linestyles="dashdot")
#     all_rate_plot.set_ylabel("ISI (ms)")
#     all_rate_plot.set_xlabel("Presynaptic Poisson Rate")
#     all_rate_plot.set_xscale("log")
#     all_rate_plot.set_yscale("log")
#
#     dir_name = f"{weight}_unbalanced_{n_trials}_trials_{sim_time}"
#     if not os.path.isdir(dir_name):
#         os.mkdir(dir_name)
#
#     all_rate_fig.savefig(f"{dir_name}/unbalanced_{n_trials}_overall_log.png")
#     # total_fig.show()


def multiple_neurons_with_poisson_neurons_unbalanced_random_rates(n_trials, sim_time, weight):
    v_r = V_REST
    tau_m = MEMBRANE_TIME_CONSTANT
    lif = """
    dv/dt = (-(v-v_r))/tau_m: 1 (unless refractory)
    I : 1 
    """
    bs.start_scope()

    tau_t = 2.2
    v_th = 1.0
    beta = 10
    threshold = "rand() < ( (1.0 / tau_t) * e**(beta * (v - v_th)))"

    reset = 'v=0'
    refractory = 1 * bs.ms

    # rates_magnitude = [1, 5, 10, 20, 50, 100, 500, 1000]
    rate_magnitudes = [100]  # poisson process rate; hz

    w = 0.1

    isi_means_each_rates = []
    isi_stds_each_rates = []

    all_trial_fig = plt.figure()
    all_trial_plot = all_trial_fig.add_subplot(111,
                                               title=f"Mean and Std for all rates for each trials (Unbalanced Input, {weight})")

    dir_name = f"siovt_unbalanced_weight_{n_trials}_trials_{sim_time}"
    rates_isi_list = []
    for rate_magnitude in rate_magnitudes:
        trial_isi_list = []

        fig = plt.figure(figsize=(12, 10))
        fig.subplots_adjust(top=0.8)

        plot1_title = "Membrane Potentials"

        u_each_trial = fig.add_subplot(311, title=plot1_title)
        spikes_each_trial = fig.add_subplot(312, sharex=u_each_trial,
                                            title=f"Spikes caused by input with rate={rate_magnitude}")
        isi_mean_std_each_trial = fig.add_subplot(313, title=f"Mean and StdDev of ISI for each neuron")
        fig.tight_layout(pad=3.0)

        isi_means_each_trials = []
        isi_stds_each_trials = []

        for trial in tqdm(range(n_trials)):
            last_length = 0

            last_spike_time = 0

            reset = 'v=0'
            refractory = 1 * bs.ms
            G = bs.NeuronGroup(N=1, model=lif, threshold=threshold, reset=reset, refractory=refractory,
                               method='euler')
            state_monitor = bs.StateMonitor(G, 'v', record=True)
            spike_monitor = bs.SpikeMonitor(G)

            ############################################################################################################
            ############################################################################################################
            ############################################################################################################
            @bs.network_operation(when="end")
            def get_isi(t):
                nonlocal last_length
                nonlocal last_spike_time
                # print(f"Time: {t}")
                # print((1.0 / tau_t) * np.e ** (beta * (G.v - v_th)))
                curr_length = spike_monitor.t.shape[0]

                # if there was a new spike, the curr length of the spike monitor would be longer than the last
                if curr_length > last_length:
                    spiked_neuron_indices = spike_monitor.i
                    spike_times = spike_monitor.t
                    for i in range(last_length, curr_length):
                        spiked_neuron_index = spiked_neuron_indices[i]
                        spike_time = spike_times[i] / bs.ms

                        # print("spike time, last_spike time", spike_time, last_spike_time)
                        curr_isi = (spike_time - last_spike_time)  # /bs.ms
                        trial_isi_list.append(curr_isi)
                        last_spike_time = spike_time
                        # print(spike_monitor.i[i], type(spike_monitor.i[i]))

                # print(last_length, curr_length)
                last_length = curr_length  # update the starting index for next set of spikes

            exc_rates = np.random.normal(5, 2.5, size=(100))
            inh_rates = np.random.normal(5, 2.5, size=(100))

            E = bs.PoissonGroup(100, rates=exc_rates * bs.Hz)
            I = bs.PoissonGroup(100, rates=inh_rates * bs.Hz)

            excitatory_connection = bs.Synapses(E, G, on_pre=f'v_post += {weight}*w')
            inhibitory_connection = bs.Synapses(I, G, on_pre='v_post -= w')

            @bs.network_operation(when="end")
            def renew_rate(t):
                nonlocal E
                nonlocal I
                # print(f"Time: {t}")
                # print((1.0 / tau_t) * np.e ** (beta * (G.v - v_th)))
                exc_rates = np.random.normal(5, 2.5, size=(100))
                inh_rates = np.random.normal(5, 2.5, size=(100))

                E = bs.PoissonGroup(100, rates=exc_rates * bs.Hz)
                I = bs.PoissonGroup(100, rates=inh_rates * bs.Hz)

                # print("renew_rate works")
            ############################################################################################################
            ############################################################################################################
            ############################################################################################################

            excitatory_connection.connect()
            inhibitory_connection.connect()

            G.v = 0.0
            bs.run(sim_time * bs.ms)

            isi_mean_var = defaultdict(tuple)
            # fig, (ax0, ax1) = plt.subplots(2, sharex=True)

            # Not considering ISI of very first spike and zero since there are some edge cases that happens
            # specifically, when the driving current is too large, the first spike occurs at 0
            # but the first "last_spike_time" is also zero
            # so for those cases, I get an ISI of 0. I just wanted to avoid that.
            # It shouldnt matter in the long run, but i didnt want that to mess up the total mean and variance
            overall_isis_for_trial = np.array(trial_isi_list[1:])
            isi_means_each_trials.append(np.mean(overall_isis_for_trial))
            isi_stds_each_trials.append(np.std(overall_isis_for_trial))

            u_each_trial.plot(state_monitor.t / bs.ms, state_monitor.v[0], label=f"{trial}")

            u_each_trial.set_xlabel('Time (ms)')
            u_each_trial.set_ylabel('v')
            # if n_neurons <= 10:
            #     ax0.legend(loc="upper right", bbox_to_anchor=(1.1, 1), ncol=2)

            trial_index = np.array([trial for _ in range(spike_monitor.t.shape[0])])

            # print(trial_index)

            spikes_each_trial.plot(spike_monitor.t / bs.ms, trial_index, 'pk')  # pk -> p = point marker, k = black
            spikes_each_trial.set_xlabel('Time (ms)')
            spikes_each_trial.set_ylabel('Neuron #')

            # neurons = []
            # means = []
            # variances = []
            # for neuron_index, mean_std in isi_mean_var.items():
            #     neurons.append(neuron_index)
            #     means.append(mean_std[0])
            #     variances.append(mean_std[1])
            isi_mean_std_each_trial.errorbar(trial, isi_means_each_trials[-1], isi_stds_each_trials[-1],
                                             linestyle="None", marker='^')
            isi_mean_std_each_trial.set_xlabel('Neuron #')
            isi_mean_std_each_trial.set_ylabel('ISI (ms)')

            # plt.show()
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            fig.savefig(f"{dir_name}/{rate_magnitude}Hz_presynaptic_spikes_{n_trials}_trials.png")

        isi_means_each_rates.append(np.mean(np.array(isi_means_each_trials)))
        isi_stds_each_rates.append(np.std(np.array(isi_stds_each_trials)))

    all_rate_fig = plt.figure()
    all_rate_plot = all_rate_fig.add_subplot(111,
                                             title=f"Mean and Std of ISI per input rates, {n_trials} trials (Unbalanced Input, {weight})")
    for i, rate in enumerate(rate_magnitudes):
        all_rate_plot.errorbar(rate, isi_means_each_rates[i], isi_stds_each_rates[i], linestyle="None", marker='^',
                               label=f"{rate}Hz Presynaptic input rate")
        all_rate_plot.text(rate, isi_means_each_rates[i], f"{rate}Hz")
    all_rate_plot.hlines(y=1, xmin=rate_magnitudes[0], xmax=rate_magnitudes[-1], linestyles="dashdot")
    all_rate_plot.set_ylabel("ISI (ms)")
    all_rate_plot.set_xlabel("Presynaptic Poisson Rate")
    all_rate_plot.set_xscale("log")
    all_rate_plot.set_yscale("log")

    dir_name = f"{weight}_unbalanced_{n_trials}_trials_{sim_time}"
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    all_rate_fig.savefig(f"{dir_name}/unbalanced_{n_trials}_overall_log.png")
    # total_fig.show()


if __name__ == '__main__':
    # multiple_neurons_with_poisson_neurons(n_neurons=5, sim_time=500)
    # multiple_neurons_with_poisson_neurons(n_neurons=10, sim_time=500)
    # multiple_neurons_with_poisson_neurons(n_neurons=100, sim_time=500)
    # multiple_neurons_with_poisson_neurons(n_neurons=200, sim_time=500)

    # multiple_neurons_with_poisson_neurons_unbalanced(n_neurons=5, sim_time=500, weight=2)
    # multiple_neurons_with_poisson_neurons_unbalanced(n_neurons=10, sim_time=500, weight=2)
    # multiple_neurons_with_poisson_neurons_unbalanced(n_neurons=100, sim_time=500, weight=2)
    # multiple_neurons_with_poisson_neurons_unbalanced(n_neurons=200, sim_time=500, weight=2)
    #

    # multiple_neurons_with_poisson_neurons_unbalanced(n_trials=20, sim_time=500, weight=10)
    multiple_neurons_with_poisson_neurons_unbalanced_random_rates(n_trials=10, sim_time=500, weight=1)
