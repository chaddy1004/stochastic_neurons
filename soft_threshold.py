import brian2 as bs
import numpy as np
from matplotlib import pyplot as plt

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


la = lambda v, tau=1.3, beta=1.0, v_th=1.0: np.random.uniform(low=0.0, high=1.0, size=1) < 1 / tau * np.exp(
    beta * (v - v_th))


def main():
    v_r = V_REST
    tau_m = MEMBRANE_TIME_CONSTANT
    lif = """
    dv/dt = ((v-v_r) + I)/tau_m: 1 (unless refractory)
    I : 1 
    """
    n_neurons = 1
    bs.start_scope()

    tau_t = 1.3
    v_th = 1.0
    beta = 4.0
    threshold = "rand() < ( (1.0 / tau_t) * 2.7182818**(beta * (v - v_th)))"

    reset = 'v=0'
    refractory = 5 * bs.ms
    G = bs.NeuronGroup(N=n_neurons, model=lif, threshold=threshold, reset=reset, refractory=refractory, method='euler')
    G.v = 0
    G.I = 5

    state_monitor = bs.StateMonitor(G, 'v', record=True)
    spike_monitor = bs.SpikeMonitor(G)

    bs.run(50 * bs.ms)

    # spike
    plt.plot(spike_monitor.t / bs.ms, spike_monitor.i, 'pk')  # the .k -> p means point marker, k means black
    plt.xlabel('Time (ms)')
    plt.ylabel('spikes')
    plt.show()

    plt.clf()
    # You can also plot the actual voltage change of each neuron if you recorded it with state monitors
    print(G.v[0])
    plt.plot(state_monitor.t / bs.ms, state_monitor.v[0])
    plt.xlabel('Time (ms)')
    plt.ylabel('v')
    plt.show()


if __name__ == '__main__':
    main()
