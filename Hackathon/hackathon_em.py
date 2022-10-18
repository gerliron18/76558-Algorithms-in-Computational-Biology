import argparse

import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.special import logsumexp
import math
from scipy.stats import beta, binom
import pandas as pd
import logomaker as logomaker


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Automata:
    alpha_high = 100
    beta_high = 1
    alpha_low = 1
    beta_low = 100
    epsilon = 0.001

    def __init__(self, transition_matrix, emission_params, num_hidden_states=3):
        self.transition_probabilities = transition_matrix
        self.alpha_high = emission_params[0, 0]
        self.beta_high = emission_params[0, 1]
        self.alpha_low = emission_params[1, 0]
        self.beta_low = emission_params[1, 1]
        self.alpha_other = emission_params[2, 0]
        self.beta_other = emission_params[2, 1]
        self.prob_meth_high = self.alpha_high / (self.alpha_high + self.beta_high)
        self.prob_meth_low = self.alpha_low / (self.alpha_low + self.beta_low)
        self.prob_meth_other = self.alpha_other / (self.alpha_other + self.beta_other)
        self.initialize_transition_probabilities()
        self.num_hidden_states = num_hidden_states

    def initialize_transition_probabilities(self):
        # self.transition_probabilities[0, 0] = self.prob_high_to_high
        # self.transition_probabilities[0, 1] = (1 - self.prob_high_to_high) / 2
        # self.transition_probabilities[0, 2] = (1 - self.prob_high_to_high) / 2
        # self.transition_probabilities[1, 1] = self.prob_low_to_low
        # self.transition_probabilities[1, 0] = (1 - self.prob_low_to_low) / 2
        # self.transition_probabilities[1, 2] = (1 - self.prob_low_to_low) / 2
        # self.transition_probabilities[2, 2] = self.prob_other_to_other
        # self.transition_probabilities[2, 0] = (1 - self.prob_other_to_other) / 2
        # self.transition_probabilities[2, 1] = (1 - self.prob_other_to_other) / 2
        with np.errstate(divide='ignore'):
            self.transition_probabilities = np.log(self.transition_probabilities)

    def get_emission_probability(self, state, num_c, num_t):

        if state == 0:  # high
            prob = binom.pmf(num_c, num_c + num_t, self.prob_meth_high)
        if state == 1:
            prob = binom.pmf(num_c, num_c + num_t, self.prob_meth_low)
        if state == 2:
            prob = binom.pmf(num_c, num_c + num_t, self.prob_meth_other)
        with np.errstate(divide='ignore'):
            return np.log(prob)

    def get_emission_probability_beta(self, state, methyl_prop):

        if state == 0:  # high

            prob = beta.cdf(x=methyl_prop + self.epsilon, a=self.alpha_high, b=self.beta_high) - beta.cdf(
                x=methyl_prop,
                a=self.alpha_high,
                b=self.beta_high)
        if state == 1:
            prob = beta.cdf(x=methyl_prop + self.epsilon, a=self.alpha_low, b=self.beta_low) - beta.cdf(
                x=methyl_prop,
                a=self.alpha_low,
                b=self.beta_low)

        with np.errstate(divide='ignore'):
            return np.log(prob)

    def get_transition_probability(self, k, l):
        return self.transition_probabilities[k, l]


def backward_alg_log_vectorized(seq, transition_matrix, emission_params):
    automata = Automata(transition_matrix, emission_params)
    backward_table = np.zeros([automata.num_hidden_states, len(seq)])
    backward_table[0, len(seq) - 1] = 1
    with np.errstate(divide='ignore'):
        backward_table = np.log(backward_table)
        # log_transition_matrix = np.log(automata.transition_probabilities)
        log_transition_matrix = automata.transition_probabilities
        # log_emission_matrix = np.log(automata.emission_probabilities)
        # log_emission_matrix = automata.emission_probabilities
    cur_seq_index = len(seq) - 2
    for letter in reversed(seq[1:]):
        future_column = backward_table[:, cur_seq_index + 1]
        log_emission_vector = np.zeros(automata.num_hidden_states)
        log_emission_vector[0] = automata.get_emission_probability(0, letter[0], letter[1] - letter[0])
        log_emission_vector[1] = automata.get_emission_probability(1, letter[0], letter[1] - letter[0])
        log_emission_vector[2] = automata.get_emission_probability(2, letter[0], letter[1] - letter[0])
        # with np.errstate(divide='ignore'):
        #     log_emission_vector = np.log(log_emission_vector)
        emission_plus_future = log_emission_vector + future_column
        emission_future_tiled = np.tile(emission_plus_future, (backward_table.shape[0], 1))
        res = emission_future_tiled + log_transition_matrix
        backward_table[:, cur_seq_index] = logsumexp(res, axis=1)
        cur_seq_index -= 1

    return backward_table, backward_table[0, 0]


def forward_alg_log_space_vectorized(seq, transition_matrix, emission_params):
    automata = Automata(transition_matrix, emission_params)
    forward_table = np.zeros([automata.num_hidden_states, len(seq)])  # begin, b1, motif, b2, end
    with np.errstate(divide='ignore'):
        forward_table = np.log(forward_table)
        # transition_probability_log_mat = np.log(automata.transition_probabilities)
        transition_probability_log_mat = automata.transition_probabilities
        # log_emission_matrix = np.log(automata.emission_probabilities)
        # log_emission_matrix = automata.emission_probabilities
    forward_table[0, 0] = 0
    cur_seq_index = 1
    for letter in seq[1:]:
        prev_f_col = forward_table[:, cur_seq_index - 1]
        prev_f_col_mat = np.tile(prev_f_col, (transition_probability_log_mat.shape[1], 1)).T
        element_multiply = prev_f_col_mat + transition_probability_log_mat
        dot_product = logsumexp(element_multiply, axis=0)
        log_emission_vector = np.zeros(automata.num_hidden_states)
        log_emission_vector[0] = automata.get_emission_probability(0, letter[0], letter[1] - letter[0])
        log_emission_vector[1] = automata.get_emission_probability(1, letter[0], letter[1] - letter[0])
        log_emission_vector[2] = automata.get_emission_probability(2, letter[0], letter[1] - letter[0])
        # with np.errstate(divide='ignore'):
        #     log_emission_vector = np.log(log_emission_vector)
        forward_table[:, cur_seq_index] = dot_product + log_emission_vector
        cur_seq_index += 1
    return forward_table, forward_table[forward_table.shape[0] - 1, forward_table.shape[1] - 1]



def viterbi_alg(seq, emission_file, p, q):
    automata = Automata(p, q, emission_file)
    v_table = np.zeros([1 + 1 + automata.motif_len + 1 + 1, len(seq)])  # begin, b1, motif, b2, end
    p_table = np.zeros([1 + 1 + automata.motif_len + 1 + 1, len(seq)])  # begin, b1, motif, b2, end
    v_table[0, 0] = 1
    with np.errstate(divide='ignore'):
        v_table = np.log(v_table)
        # transition_probability_log_mat = np.log(automata.transition_probabilities)
        log_emission_matrix = np.log(automata.emission_probabilities)
        transition_probability_log_mat = automata.transition_probabilities
        # log_emission_matrix = automata.emission_probabilities
    cur_seq_index = 1
    for letter in seq[1:]:
        letter_ind = automata.letter_to_ind_map[letter]
        prev_v_col_mat = np.tile(v_table[:, cur_seq_index - 1], (transition_probability_log_mat.shape[1], 1)).T
        prev_c_col_transition_prod = prev_v_col_mat + transition_probability_log_mat
        argmax_vals = [int(x) for x in np.argmax(prev_c_col_transition_prod, axis=0)]
        max_vals = prev_c_col_transition_prod.T[np.arange(len(prev_c_col_transition_prod)), argmax_vals]
        v_table[:, cur_seq_index] = max_vals + log_emission_matrix[:, letter_ind]
        p_table[:, cur_seq_index] = np.array(argmax_vals)
        cur_seq_index += 1
    cur_state = p_table.shape[0] - 2
    res_list = []
    min_motif_loc = -1
    for i in reversed(range(1, len(seq) - 1)):
        if cur_state == 1 or cur_state == p_table.shape[0] - 2:
            res_list.append('B')
        else:
            res_list.append('M')
            min_motif_loc = i - 1
        cur_state = p_table[int(cur_state), i]
    # print_alignments("".join(res_list), list(reversed(seq[1:-1])))
    return min_motif_loc


def state_to_state_str(state):
    if state == 0:
        return f"{bcolors.RED}H{bcolors.ENDC}"
    elif state == 1:
        return f"{bcolors.OKGREEN}L{bcolors.ENDC}"
    elif state == 2:
        return f"{bcolors.OKBLUE}O{bcolors.ENDC}"


def e_m(p, q, r, in_seq_list, inseq_list_counts, convergence_threshold):
    ll_history = []
    cur_gradient = 1000

    num_hidden_states = 3
    initial_state_emission_counts = np.zeros([num_hidden_states, 2])
    initial_state_emission_counts[0, 0] = 10
    initial_state_emission_counts[0, 1] = 1

    initial_state_emission_counts[1, 1] = 10
    initial_state_emission_counts[1, 0] = 1

    initial_state_emission_counts[2, 1] = 5
    initial_state_emission_counts[2, 0] = 5

    transition_probabilities = np.zeros([num_hidden_states, num_hidden_states])
    transition_probabilities[0, 0] = p
    transition_probabilities[0, 1] = (1 - p) / 2
    transition_probabilities[0, 2] = (1 - p) / 2
    transition_probabilities[1, 1] = q
    transition_probabilities[1, 0] = (1 - q) / 2
    transition_probabilities[1, 2] = (1 - q) / 2
    transition_probabilities[2, 2] = r
    transition_probabilities[2, 0] = (1 - r) / 2
    transition_probabilities[2, 1] = (1 - r) / 2

    est_transition_matrix, emission_params, logliklihood = e_m_iteration(in_seq_list, inseq_list_counts,
                                                                         transition_probabilities,
                                                                         initial_state_emission_counts)
    ll_history.append(logliklihood)
    print(logliklihood)
    prev_logliklihood = -100000000000
    while logliklihood - prev_logliklihood > convergence_threshold:
        prev_logliklihood = logliklihood
        est_transition_matrix, emission_params, logliklihood = e_m_iteration(in_seq_list, inseq_list_counts,
                                                                             est_transition_matrix, emission_params)
        ll_history.append(logliklihood)
        print(logliklihood)
    in_seq_1 = inseq_list_counts[0]
    f_table, _ = forward_alg_log_space_vectorized(in_seq_1, est_transition_matrix, emission_params)
    b_table, _ = backward_alg_log_vectorized(in_seq_1, est_transition_matrix, emission_params)
    posterior_table = f_table + b_table
    argmax_vector = np.argmax(posterior_table, axis=0)
    methylation_percents = []
    methylation_counts = []
    states = []
    for i in range(f_table.shape[1]):
        methylation_percent = round(round((in_seq_1[i][0] / in_seq_1[i][1]), 2) * 100)
        methylation_percents.append(methylation_percent)
        methylation_counts.append(str(in_seq_1[i][1]))
        states.append(argmax_vector[i])
        # if len(states) >= 50:
        #     percents_string = "\t".join(methylation_percents)
        #     counts_string = "\t".join(methylation_counts)
        #     states_string = "\t".join(states)
        #     # print(percents_string)
        #     # print(counts_string)
        #     # print(states_string)
        #     # print()
        #     methylation_percents = []
        #     methylation_counts = []
        #     states = []
    # percents_string = "\t".join(methylation_percents)
    # counts_string = "\t".join(methylation_counts)
    # states_string = "\t".join(states)
    print(methylation_percents)
    print(states)
    colors = []
    for state in states:
        if state == 0:
            colors.append('red')
        if state == 1:
            colors.append('green')
        if state == 2:
            colors.append('blue')
    # plt.plot(np.arange(len(states)), methylation_percents, linewidth=1.2, color='k')
    plt.scatter(np.arange(len(states)), methylation_percents, color=colors)
    plt.show()
    return pd.DataFrame(data=np.exp(posterior_table).T, columns=["H", "L", "O"])
    # print(percents_string)
    # print(counts_string)
    # print(states_string)




def e_m_iteration(in_seq_list_percents, in_seq_lis_counts, transition_matrix, emission_params):
    automata = Automata(transition_matrix, emission_params)
    state_transfer_counts = np.zeros(
        [automata.num_hidden_states, automata.num_hidden_states])
    state_emission_counts = np.zeros([automata.num_hidden_states, 2])
    with np.errstate(divide='ignore'):
        state_transfer_counts = np.log(state_transfer_counts)
        state_emission_counts = np.log(state_emission_counts)
    log_liklihood_sum = 0
    for in_seq in in_seq_lis_counts:
        f_table, _ = forward_alg_log_space_vectorized(in_seq, transition_matrix, emission_params)
        b_table, _ = backward_alg_log_vectorized(in_seq, transition_matrix, emission_params)

        p_x = logsumexp(f_table[:, -1])
        log_liklihood_sum += p_x
        with np.errstate(divide='ignore'):
            sum_over_seq = np.log(0)
        for state_ind_1 in range(automata.transition_probabilities.shape[0]):
            for state_ind_2 in range(automata.transition_probabilities.shape[0]):
                with np.errstate(divide='ignore'):
                    sum_over_seq = np.log(0)
                for i in range(0, f_table.shape[1]):
                    emission_prob = automata.get_emission_probability(state_ind_2, in_seq[i][0], in_seq[i][1] - in_seq[i][0])
                    p_k_l = (f_table[state_ind_1, i - 1] + automata.transition_probabilities[state_ind_1, state_ind_2] +
                              emission_prob + b_table[
                                 state_ind_2, i]) - p_x
                    sum_over_seq = logsumexp([p_k_l, sum_over_seq])
                state_transfer_counts[state_ind_1, state_ind_2] = logsumexp(
                    [sum_over_seq, state_transfer_counts[state_ind_1, state_ind_2]])
        for in_seq_count in in_seq_lis_counts:
            for state_ind_1 in range(automata.transition_probabilities.shape[0]):
                with np.errstate(divide='ignore'):
                    sum_over_seq_meth = np.log(0)
                    sum_over_seq_unmeth = np.log(0)
                for i in range(f_table.shape[1]):
                    posterior_prob = (f_table[state_ind_1, i] + b_table[state_ind_1, i]) - p_x
                    with np.errstate(divide='ignore'):
                        log_count_c = np.log(in_seq_count[i][0])
                        log_count_t = np.log((in_seq_count[i][1] - in_seq_count[i][0]))
                    posterior_times_meth = posterior_prob + log_count_c
                    posterior_times_unmeth = posterior_prob + log_count_t
                    sum_over_seq_meth = logsumexp(
                        [posterior_times_meth, sum_over_seq_meth])
                    sum_over_seq_unmeth = logsumexp(
                        [posterior_times_unmeth, sum_over_seq_unmeth])
                state_emission_counts[state_ind_1, 0] = logsumexp(
                    [state_emission_counts[state_ind_1, 0], sum_over_seq_meth])
                state_emission_counts[state_ind_1, 1] = logsumexp(
                    [state_emission_counts[state_ind_1, 1], sum_over_seq_unmeth])
    est_transition_probs = np.zeros([automata.num_hidden_states, automata.num_hidden_states])
    state_transfer_counts = np.exp(state_transfer_counts)
    for state_ind_1 in range(automata.transition_probabilities.shape[0]):
        for state_ind_2 in range(automata.num_hidden_states):
            p_numerator = state_transfer_counts[state_ind_1, state_ind_2]
            p_denominator_entries = state_transfer_counts[state_ind_1, :]
            p_denominator = sum(p_denominator_entries)
            if p_numerator == 0:
                est_p = 0.01
            else:
                est_p = p_numerator / p_denominator
            est_transition_probs[state_ind_1, state_ind_2] = est_p

    # print("q: " + str(est_q))
    state_emission_ests = np.exp(state_emission_counts)
    state_emission_ests += 1
    print(est_transition_probs)
    print(state_emission_ests)
    return est_transition_probs, state_emission_ests, log_liklihood_sum
    # return force_known_emissions(automata.emission_probabilities, automata.motif_len), est_p, est_q, log_liklihood_sum


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--alg', help='Algorithm (e.g. viterbi)', required=True)
#     parser.add_argument('seq', help='A sequence over the alphabet [A,C,G,T] (e.g. ACTGGACTACGTCATGCA)')
#     parser.add_argument('initial_emission', help='Path to emission table (e.g. initial_emission.tsv)')
#     parser.add_argument('p', help='transition probability p (e.g. 0.01)', type=float)
#     parser.add_argument('q', help='transition probability q (e.g. 0.5)', type=float)
#     args = parser.parse_args()
#
#     in_sequence = "^" + args.seq + "$"
#
#     if args.alg == 'viterbi':
#         viterbi_alg(in_sequence, args.initial_emission, args.p, args.q)
#
#     elif args.alg == 'forward':
#         _, logliklihood = forward_alg_log_space_vectorized(in_sequence, args.initial_emission, args.p, args.q)
#         print(logliklihood)
#
#     elif args.alg == 'backward':
#         _, logliklihood = backward_alg_log_vectorized(in_sequence, args.initial_emission, args.p, args.q)
#         print(logliklihood)
#
#     elif args.alg == 'posterior':
#         forward_log_table, _ = forward_alg_log_space_vectorized(in_sequence, args.initial_emission, args.p, args.q)
#         backward_log_table, _ = backward_alg_log_vectorized(in_sequence, args.initial_emission, args.p, args.q)
#         most_likely_states(forward_log_table, backward_log_table, in_sequence)
#
#

def parse_args():
    """
    Parse the command line arguments.
    :return: The parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', help='File path with list of sequences (e.g. seqs_ATTA.fasta)',
                        default="seqs_ATTA.fasta")
    parser.add_argument('--seed', help='Guess for the motif (e.g. ATTA)', default="ATTA")
    parser.add_argument('--p', type=float, help='Initial guess for the p transition probability (e.g. 0.01)',
                        default=0.85)
    parser.add_argument('--q', type=float, help='Initial guess for the q transition probability (e.g. 0.9)',
                        default=0.7)
    parser.add_argument('--t', type=float, help='Initial guess for the q transition probability (e.g. 0.9)',
                        default=0.1)
    parser.add_argument('--alpha', type=float, help='Softening parameter for the initial profile (e.g. 0.1)',
                        default=0.7)
    parser.add_argument('--convergenceThr', type=float, help='ll improvement threshold for the stopping condition'
                                                             ' (e.g. 0.1)', default=0.1)
    return parser.parse_args()


def eliminate_zeros_add_one(arr):
    arr[:, 0] += 1
    arr[:, 1] += 2
    return arr


def create_seq_logo(df):
    # create Logo object
    crp_logo = logomaker.Logo(df,color_scheme={"H": "red","O": "blue", "L": "green"}, font_name='Arial Rounded MT Bold')

    # style using Logo methods
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    crp_logo.style_xticks(rotation=90, fmt='%d', anchor=0)

    # style using Axes methods
    crp_logo.ax.set_ylabel("Probability", labelpad=-1)
    crp_logo.ax.xaxis.set_ticks_position('none')
    crp_logo.ax.xaxis.set_tick_params(pad=-1)

    # style and show figure
    crp_logo.fig.show()


def main():
    args = parse_args()
    # fastas = []
    # with open(args.fasta) as fastas_file:
    #     lines = fastas_file.readlines()
    # fasta_pattern = re.compile('^\s*([ACGT]*)\s*$')
    # for line in lines:
    #     fasta_line = fasta_pattern.match(line)
    #     if fasta_line:
    #         fastas.append('^' + fasta_line.group(1) + '$')
    # cpg_start = 16812916
    # cpg_end = 16823087
    cpg_start = 16823087 - 100
    cpg_end = 16823087
    filename = 'Data/Prostate-Epithelial-Z000000S3.beta'
    arr = np.fromfile(filename, dtype=np.uint8).reshape((-1, 2))
    arr = arr[cpg_start:cpg_end, :]
    arr = eliminate_zeros_add_one(arr)
    fastas = [[0.99, 0.99, 0.95, 0.94, 0.95, 0.9, 0.9, 0.95, 0.01, 0.01, 0.1, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]]
    in_seq_counts = [[[99, 100], [99, 100], [95, 100], [94, 100], [95, 100], [90, 100], [90, 100], [95, 100], [1, 100], [1, 100], [10, 100], [5, 100], [1, 100], [1, 100], [1, 100], [1, 100], [1, 100]]]
    fastas = [[0.99, 0.99, 0.95, 0.94, 0.95, 0.9, 0.9, 0.95]]
    in_seq_counts = [
        arr]
    df = e_m(args.p, args.q, args.t, fastas, in_seq_counts, args.convergenceThr)
    create_seq_logo(df)

if __name__ == '__main__':
    main()
