import random
import numpy as np
import pandas as pd
from collections import defaultdict
import networkx as nx
import json

class MarkovModel:
    def __init__(self, state_size=1):
        self.state_size = state_size
        self.states = defaultdict(int)
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.graph = nx.DiGraph()
        self.max_state_probs = dict()
        # global symbol vocabulary (next-state support for smoothing)
        self.vocab = set()


    def save_json(self, filepath):
        """Saves the trained model to a human-readable JSON file."""
        # Helper to convert tuple keys to string keys
        def convert_keys_to_str(d):
            return {'|'.join(k): v for k, v in d.items()}

        model_data = {
            'state_size': self.state_size,
            'states': convert_keys_to_str(self.states),
            'transitions': {
                '|'.join(k): v for k, v in self.transitions.items()
            },
            'max_state_probs': convert_keys_to_str(self.max_state_probs)
        }

        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=4)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_json(cls, filepath):
        """Loads a model from a JSON file."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        # Helper to convert string keys back to tuple keys
        def convert_keys_to_tuple(d):
            return {tuple(k.split('|')): v for k, v in d.items()}

        state_size = model_data['state_size']
        model = cls(state_size=state_size)

        # Load states
        loaded_states = convert_keys_to_tuple(model_data['states'])
        model.states.update(loaded_states)

        # Load transitions
        loaded_transitions = {
            tuple(k.split('|')): v for k, v in model_data['transitions'].items()
        }
        for state, next_states in loaded_transitions.items():
            model.transitions[state].update(next_states)
            
        # Load max state probabilities
        loaded_max_probs = convert_keys_to_tuple(model_data['max_state_probs'])
        model.max_state_probs.update(loaded_max_probs)

        # Rebuild the graph from the loaded transitions
        vocab = set()
        for state, next_states in model.transitions.items():
            for token in state:
                vocab.add(token)
            for next_state, count in next_states.items():
                graph_next_state = tuple(list(state)[1:] + [next_state])
                model.graph.add_edge(state, graph_next_state, weight=count)
                vocab.add(next_state)
        model.vocab = vocab
        
        print(f"Model loaded from {filepath}")
        return model

    def train(self, corpus):
        for sentence in corpus:
            words = sentence.split()
            for w in words:
                self.vocab.add(w)
            for i in range(len(words) - self.state_size):
                state = tuple(words[i:i+self.state_size]) 
                next_state = words[i+self.state_size]
                self.states[state] += 1
                self.transitions[state][next_state] += 1
                graph_next_state = tuple(words[i+1:i+self.state_size+1])
                self.graph.add_edge(state, graph_next_state, weight=self.transitions[state][next_state])
        
        for state, next_states in self.transitions.items():
            state_count = self.states[state]
            if state_count > 0:
                max_prob = max(count / state_count for count in next_states.values())
                self.max_state_probs[state] = max_prob

    def move(self, state):
        """Sample the next symbol from a given state (context).
        Returns None if the state has no outgoing transitions."""
        next_counts = self.transitions.get(state, {})
        if not next_counts:
            return None
        next_states = list(next_counts.keys())
        weights = list(next_counts.values())
        return random.choices(next_states, weights=weights, k=1)[0]

    def generate(self, length, state=0):
        """Generate a sequence of the given length.
        - Seeds from a valid state (with outgoing transitions) if none/invalid provided.
        - Re-seeds if a dead-end state is encountered during generation.
        """
        # Collect states with outgoing transitions
        valid_states = [s for s, nxt in self.transitions.items() if len(nxt) > 0]
        if not valid_states:
            raise ValueError("No valid states with outgoing transitions. Train the model with more data.")

        # Normalize provided seed
        provided = None
        if state != 0 and state is not None:
            provided = tuple(state) if isinstance(state, (list, tuple)) else (state,)

        if provided is None or not self.transitions.get(provided):
            current_state = random.choice(valid_states)
        else:
            current_state = provided

        generated_sequence = list(current_state)
        # Ensure sequence has at least state_size tokens
        if len(generated_sequence) < self.state_size:
            pad_state = random.choice(valid_states)
            generated_sequence = list(pad_state)

        while len(generated_sequence) < max(length, self.state_size + 1):
            ctx = tuple(generated_sequence[-self.state_size:]) if self.state_size > 0 else tuple()
            next_word = self.move(ctx)
            if next_word is None:
                # Dead end: reseed with another valid context
                ctx = random.choice(valid_states)
                # Append its tokens to keep momentum, then continue
                generated_sequence.extend(list(ctx))
                # Trim to avoid growing too fast
                generated_sequence = generated_sequence[-(self.state_size + 1):]
                continue
            generated_sequence.append(next_word)

        return ' '.join(generated_sequence[:length])

    def likelihood(self, sequence):
        words = sequence.split()
        likelihood = 1
        for i in range(len(words) - self.state_size):
            state = tuple(words[i:i+self.state_size]) 
            next_state = words[i+self.state_size]
            transition_count = self.transitions[state][next_state]
            state_count = self.states[state]
            likelihood *= transition_count / state_count if state_count > 0 else 0
        return likelihood

    def likelihood_with_smoothing(self, sequence, alpha=1.0):
        """
        Additive (Laplace) smoothing for transition probabilities:
        P = (count(state→next) + alpha) / (count(state) + alpha * V)
        where V is the number of possible next symbols (global vocab size).
        """
        words = sequence.split()
        likelihood = 1.0
        V = max(1, len(self.vocab))  # avoid divide-by-zero
        for i in range(len(words) - self.state_size):
            state = tuple(words[i:i+self.state_size])
            next_state = words[i+self.state_size]
            if state in self.states:
                state_count = self.states[state]
                transition_count = self.transitions.get(state, {}).get(next_state, 0)
            else:
                state_count = 0
                transition_count = 0
            prob = (transition_count + alpha) / (state_count + alpha * V)
            likelihood *= prob
        return likelihood

    def perplexity(self, sequence, smoothing_alpha=1.0, eps=1e-12):
        """
        Return a list of per-step transition probabilities (or log-probs) for the sequence.
        For tokens w0 w1 ... wT and state_size = k, this returns probabilities for
        each transition: P(w_{i+k} | w_i ... w_{i+k-1}), i = 0 .. T-k-1.
        If smoothing_alpha > 0, uses Laplace smoothing.
        """
        words = sequence.split()
        logprobs = []
        V = max(1, len(self.vocab))  # for smoothing
        if len(words) <= self.state_size:
            return logprobs  # nothing to compute
        for i in range(len(words) - self.state_size):
            state = tuple(words[i:i + self.state_size])
            nxt = words[i + self.state_size]
            state_count = self.states.get(state, 0)
            trans_count = self.transitions.get(state, {}).get(nxt, 0)
            # Apply smoothing
            if smoothing_alpha > 0:
                p = (trans_count + smoothing_alpha) / (state_count + smoothing_alpha * V)
            else:
                p = (trans_count / state_count) if state_count > 0 else 0.0
            logprobs.append(np.log(p + eps))
        nll = -sum(logprobs)
        ppl = np.exp(nll / max(1, len(logprobs)))
        return ppl

    def step_distributions(self, sequence, alpha=1.0, log=False, backoff='unigram', order='vocab'):
        """
        Return a 2D array of shape [T, V] with P(next_token | state_t) for each time step t,
        where T = len(sequence.split()) - state_size and V = |vocab|.
        Also returns token_order (list of tokens corresponding to columns).
        - alpha: Laplace smoothing.
        - backoff: 'unigram' (global next-token counts) or 'uniform' for unseen states.
        - order: 'vocab' (sorted tokens) or 'freq' (descending unigram frequency).
        """
        words = sequence.split()
        V = len(self.vocab)
        if V == 0 or len(words) <= self.state_size:
            return np.zeros((0, 0), dtype=np.float64), []

        # Choose a stable token order for columns
        if order == 'freq':
            uni = {}
            for ns in self.transitions.values():
                for tok, c in ns.items():
                    uni[tok] = uni.get(tok, 0) + c
            token_order = [t for t, _ in sorted(uni.items(), key=lambda x: (-x[1], x[0]))]
            token_order += [t for t in sorted(self.vocab) if t not in token_order]
        else:
            token_order = sorted(self.vocab)
        idx = {t: i for i, t in enumerate(token_order)}

        # Precompute unigram backoff counts
        if backoff == 'unigram':
            back_counts = np.zeros(V, dtype=np.float64)
            for ns in self.transitions.values():
                for tok, c in ns.items():
                    back_counts[idx[tok]] += c
            back_total = back_counts.sum()
        else:
            back_counts = None
            back_total = 0.0

        T = len(words) - self.state_size
        out = np.zeros((T, V), dtype=np.float64)

        for t in range(T):
            state = tuple(words[t:t + self.state_size])
            if state in self.states and self.states[state] > 0:
                denom = self.states[state] + alpha * V
                row = np.full(V, alpha / denom, dtype=np.float64)  # unseen-next baseline
                for tok, c in self.transitions[state].items():
                    row[idx[tok]] = (c + alpha) / denom
            else:
                if backoff == 'unigram' and back_total > 0:
                    denom = back_total + alpha * V
                    row = (back_counts + alpha) / denom
                else:
                    row = np.full(V, 1.0 / V, dtype=np.float64)

            out[t] = np.log(row + 1e-12) if log else row

        return out, token_order

    def predicted_ranks(self, sequence, alpha=1.0, backoff='unigram', order='vocab'):
        """
        Compute the rank (1 = best) of the ground-truth next token across the trajectory.
        Uses step_distributions (probabilities, not logs). Returns a list of ranks.
        """
        # Get per-step distributions
        result = self.step_distributions(sequence, alpha=alpha, log=False, backoff=backoff, order=order)
        if not isinstance(result, tuple) or len(result) != 2:
            return 0.0
        probs, token_order = result
        if probs.size == 0:
            return 0.0
        idx = {t: i for i, t in enumerate(token_order)}

        words = sequence.split()
        T = len(words) - self.state_size
        if T <= 0:
            return 0.0

        ranks = []
        for t in range(T):
            true_next = words[t + self.state_size]
            p_row = probs[t]
            if true_next in idx:
                p_true = p_row[idx[true_next]]
                # rank = 1 + number of tokens with strictly greater probability
                rank = 1 + int((p_row > p_true).sum())
            else:
                # unseen token treated as worst rank
                rank = p_row.shape[0]
            ranks.append(float(rank))
        
        return ranks


    def mean_true_rank(self, sequence, alpha=1.0, backoff='unigram', order='vocab'):
        ranks = self.predicted_ranks(sequence, alpha=alpha, backoff=backoff, order=order)
        return float(np.mean(ranks)) if ranks else 0.0
    
    def topk_accuracy_single(self, sequence, k=10, alpha=1.0, backoff='unigram', order='vocab'):
        ranks = self.predicted_ranks(sequence, alpha=alpha, backoff=backoff, order=order)
        hits = sum(1 for r in ranks if r <= k)
        return float(hits / len(ranks)) if len(ranks) > 0 else 0.0

    def topk_accuracy(self, sequence, alpha=1.0, backoff='unigram', order='vocab', k_values=[1,5,10]):
        accuracies = {}
        for k in k_values:
            acc = self.topk_accuracy_single(sequence, k=k, alpha=alpha, backoff=backoff, order=order)
            accuracies[k] = acc
        return accuracies

    def topk_accuracy_single_fast(self, sequence, k=10, alpha=1.0, backoff='unigram', order='vocab'):
        """
        Compute top-k accuracy for a single trajectory: fraction of steps where
        the ground-truth next token is among the model's top-k predictions.
        Returns a single float in [0, 1].
        """
        # Get per-step distributions
        result = self.step_distributions(sequence, alpha=alpha, log=False, backoff=backoff, order=order)
        if not isinstance(result, tuple) or len(result) != 2:
            return 0.0
        probs, token_order = result
        if probs.size == 0:
            return 0.0
        idx = {t: i for i, t in enumerate(token_order)}

        words = sequence.split()
        T = len(words) - self.state_size
        if T <= 0:
            return 0.0

        hits = 0
        total = 0
        V = probs.shape[1]
        kk = min(max(1, int(k)), V)
        for t in range(T):
            true_next = words[t + self.state_size]
            p_row = probs[t]
            if true_next not in idx:
                total += 1
                continue
            true_idx = idx[true_next]
            # Find top-k indices (descending). Use argpartition for efficiency.
            if kk < V:
                topk_idx = np.argpartition(-p_row, kk - 1)[:kk]
                # ensure exact order not required for membership test
            else:
                topk_idx = np.arange(V)
            if true_idx in topk_idx:
                hits += 1
            total += 1

        return float(hits / total) if total > 0 else 0.0

    
    def geometric_mean_likelihood(self, sequence):
        words = sequence.split()
        likelihoods = []
        for i in range(len(words) - self.state_size):
            state = tuple(words[i:i+self.state_size]) 
            next_state = words[i+self.state_size]
            transition_count = self.transitions.get(state, {}).get(next_state, 0)
            state_count = self.states.get(state, 0)
            if state_count > 0:
                likelihoods.append(transition_count / state_count)
        if likelihoods:
            return np.prod(likelihoods) ** (1 / len(likelihoods))
        else:
            return 0
    
    def weighted_double_mean_likelihood(self, sequence, alpha=0.5):
        words = sequence.split()
        valid_transitions = 0
        valid_states = 0
        total_possible_transitions = len(words) - self.state_size
        total_possible_states = len(words) - self.state_size + 1
        
        # Lists to store log probabilities for valid transitions and states
        transition_log_probs = []
        state_log_probs = []
        
        for i in range(total_possible_transitions):
            state = tuple(words[i:i+self.state_size])
            next_state = words[i+self.state_size]
            
            # Valid transition check
            if next_state in self.transitions.get(state, {}):
                transition_count = self.transitions[state][next_state]
                state_count = self.states[state]
                if state_count > 0:
                    transition_prob = transition_count / state_count
                    transition_log_probs.append(np.log(transition_prob))
                    valid_transitions += 1
            
            # Valid state check
            if state in self.states:
                state_prob = self.states[state] / sum(self.states.values())
                state_log_probs.append(np.log(state_prob))
                valid_states += 1

        # Calculate the geometric means
        if valid_transitions > 0:
            mean_transition_likelihood = np.exp(np.sum(transition_log_probs) / valid_transitions)
        else:
            mean_transition_likelihood = 0
            
        if valid_states > 0:
            mean_state_likelihood = np.exp(np.sum(state_log_probs) / valid_states)
        else:
            mean_state_likelihood = 0

        # Calculate transition_ratio and state_ratio
        transition_ratio = valid_transitions / total_possible_transitions if total_possible_transitions > 0 else 0
        state_ratio = valid_states / total_possible_states if total_possible_states > 0 else 0

        # Return weighted combination
        return (1 - alpha) * transition_ratio * mean_transition_likelihood + alpha * state_ratio * mean_state_likelihood

        
    def weighted_geometric_mean_filtered_likelihood(self, sequence, alpha=0.5):
        words = sequence.split()
        likelihoods = []
        valid_transitions = 0
        total_possible_transitions = len(words) - self.state_size
        
        for i in range(total_possible_transitions):
            state = tuple(words[i:i+self.state_size])
            next_state = words[i+self.state_size]
            if next_state in self.transitions.get(state, {}):
                transition_count = self.transitions[state][next_state]
                state_count = self.states[state]
                if state_count > 0:
                    likelihoods.append(transition_count / state_count)
                    valid_transitions += 1
        
        if likelihoods:
            geometric_mean = np.prod(likelihoods) ** (1 / len(likelihoods))
            transition_ratio = valid_transitions / total_possible_transitions
            return alpha * geometric_mean + (1 - alpha) * transition_ratio
        else:
            return 0
        

    def max_normalized_transition_likelihood(self, sequence, alpha=0.5):
        """
        Computes the likelihood of a sequence based on max-normalized transition probabilities.

        - If a transition does not exist in training, its score is 0.
        - If a transition exists, its score is normalized between 0 and 1,
        where 1 corresponds to the most frequent transition from that state.

        Returns:
            - A likelihood score between 0 and 1.
        """
        words = sequence.split()
        transition_scores = []
        total_possible_transitions = len(words) - self.state_size

        if total_possible_transitions <= 0:
            return 0  # Sequence too short

        for i in range(total_possible_transitions):
            state = tuple(words[i:i+self.state_size])
            next_state = words[i+self.state_size]

            if state in self.states:
                state_transitions = self.transitions.get(state, {})
                if next_state in self.transitions.get(state, {}):
                    transition_count = self.transitions[state][next_state]
                    # Compute transition probability
                    state_count = self.states[state]
                    transition_prob = transition_count / state_count
                    # Max-normalization: Normalize by the most frequent transition from this state
                    max_transition_prob = self.max_state_probs.get(state, 1.0) 
                    normalized_score = transition_prob / max_transition_prob  # Normalize to [0,1]
                    # Scale between alpha and 1
                    scaled_score = alpha + (1-alpha) * normalized_score
                else:
                    scaled_score =0 
            else:
                scaled_score = 0  # If state was never seen, assign lowest likelihood
            transition_scores.append(scaled_score)

        # Compute average of normalized transition scores
        sequence_likelihood = np.mean(transition_scores) if transition_scores else 0

        return sequence_likelihood
    
    def powered_max_norm_likelihood(self, sequence, gamma=1.0):
        """
        Computes the likelihood of a sequence based on max-normalized transition probabilities,
        with an optional exponent to control the contrast between frequent and rare transitions.

        Parameters:
            sequence (str): The input sequence as a string of space-separated states.
            gamma (float): Exponent applied to the normalized score. Higher gamma sharpens contrast.

        Returns:
            float: Likelihood score in [0, 1], averaged across all transitions.
        """
        words = sequence.split()
        transition_scores = []
        total_possible_transitions = len(words) - self.state_size

        if total_possible_transitions <= 0:
            return 0  # Sequence too short

        for i in range(total_possible_transitions):
            state = tuple(words[i:i + self.state_size])
            next_state = words[i + self.state_size]

            if state in self.states:
                state_transitions = self.transitions.get(state, {})
                if next_state in state_transitions:
                    transition_count = self.transitions[state][next_state]
                    state_count = self.states[state]
                    transition_prob = transition_count / state_count

                    # Normalize by max transition from this state
                    max_transition_prob = self.max_state_probs.get(state, 1.0)
                    normalized_score = transition_prob / max_transition_prob  # In [0, 1]


                    scaled_score = normalized_score ** gamma
                else:
                    scaled_score = 0
            else:
                scaled_score = 0

            transition_scores.append(scaled_score)

        return np.mean(transition_scores) if transition_scores else 0

    

    def likelihood_ratio_vs_random(self, sequence, epsilon=1e-6):
        """
        Computes the likelihood ratio of a sequence compared to a uniform random baseline.
        
        Returns:
            - Likelihood ratio: how much more likely the sequence is compared to random.
        """
        words = sequence.split()
        log_likelihoods = []
        total_possible_transitions = len(words) - self.state_size

        if total_possible_transitions <= 0:
            return 0  # Sequence too short

        num_possible_transitions = len(self.transitions)  # Total number of possible transitions

        for i in range(total_possible_transitions):
            state = tuple(words[i:i+self.state_size])
            next_state = words[i+self.state_size]

            # Compute transition probability in trained model
            if state in self.states:
                state_count = self.states[state]
                transition_count = self.transitions.get(state, {}).get(next_state, 0)
                transition_prob = (transition_count + epsilon) / (state_count + epsilon * num_possible_transitions)
            else:
                transition_prob = epsilon  # Almost zero for unseen transitions

            log_likelihoods.append(np.log(transition_prob))

        # Compute sequence likelihood using the Markov model
        avg_log_likelihood = np.sum(log_likelihoods) / total_possible_transitions
        likelihood_model = np.exp(avg_log_likelihood)

        # Compute uniform random probability using the same log-formulation
        uniform_transition_prob = 1 / num_possible_transitions if num_possible_transitions > 0 else epsilon
        log_random_prob = np.log(uniform_transition_prob)
        likelihood_random = np.exp(log_random_prob)  # Keeping it in the same formulation

        # Compute likelihood ratio
        likelihood_ratio = likelihood_model / likelihood_random if likelihood_random > 0 else 0

        return likelihood_ratio


    def state_normalized_likelihood(self, sequence):
        """
        Computes the likelihood of a sequence based on transition probabilities 
        considering hour transitions (intra-hour and inter-hour).
        
        - Normalized by total transitions from the given state, not the total for the hour.
        - Returns only the mean likelihood across transitions.
        
        Returns:
            - mean_likelihood: average likelihood across transitions
        """
        words = sequence.split()
        transition_probs = []
        total_possible_transitions = len(words) - self.state_size

        if total_possible_transitions <= 0:
            return 0  # Sequence too short

        for i in range(total_possible_transitions):
            state = tuple(words[i:i+self.state_size])
            next_state = words[i+self.state_size]
            # Get all transitions that occurred from this state (state -> any next state)
            total_state_transitions = sum(self.transitions.get(state, {}).values())

            if total_state_transitions > 0:
                # Compute transition probability within this hour pair segment
                transition_count = self.transitions.get(state, {}).get(next_state, 0)
                transition_prob = transition_count / total_state_transitions
            else:
                # If no transitions from this state exist, assign low probability
                transition_prob = 0

            transition_probs.append(transition_prob)

        # Compute the mean likelihood
        mean_likelihood = np.mean(transition_probs) if transition_probs else 0

        return mean_likelihood
    


    def log_likelihood(self, sequence, epsilon=1e-10):
        words = sequence.split()
        log_likelihoods = []
        total_possible_transitions = len(words) - self.state_size

        for i in range(total_possible_transitions):
            state = tuple(words[i:i+self.state_size])
            next_state = words[i+self.state_size]

            # Check if the state was seen before
            if state in self.states:
                state_count = self.states[state]
                transition_count = self.transitions.get(state, {}).get(next_state, 0)

                # Compute transition probability with Laplace smoothing
                transition_prob = (transition_count + epsilon) / (state_count + epsilon * len(self.states))
            else:
                # If state has never been seen, assign a very low probability
                transition_prob = epsilon  # Almost 0 but avoids log(0)

            log_likelihoods.append(np.log(transition_prob))
            print(transition_prob)

        # Compute average log-likelihood
        avg_log_likelihood = np.sum(log_likelihoods) / total_possible_transitions

        # Exponentiation to get a likelihood score between 0 and 1
        return -avg_log_likelihood
    

    def log_likelihood_emphasize_frequent(self, sequence, epsilon=1e-10):
        """
        Computes a likelihood score where frequent transitions (relative to their origin state)
        are emphasized using a log-based penalty of their inverse normalized form.

        The score is:
            1 - avg_t [ log(1 - normalized_prob + ε) ]
        """
        words = sequence.split()
        log_penalties = []
        total_possible_transitions = len(words) - self.state_size

        if total_possible_transitions <= 0:
            return 0

        for i in range(total_possible_transitions):
            state = tuple(words[i:i + self.state_size])
            next_state = words[i + self.state_size]

            if state in self.states:
                state_transitions = self.transitions.get(state, {})
                state_count = self.states[state]

                if next_state in state_transitions:
                    transition_count = state_transitions[next_state]
                    transition_prob = transition_count / state_count
                
                    penalty = np.log(1 - transition_prob + epsilon)
                else:
                    # If transition never seen, treat as max penalty
                    penalty = np.log(1 + epsilon)  # log(1) = 0, still contributes nothing
            else:
                penalty = np.log(1 + epsilon)

            log_penalties.append(penalty)

        avg_penalty = np.exp(np.mean(log_penalties))
        score = 1 - abs(avg_penalty)  # flip to make high score = likely

        return score

    

    
    def transition_count_likelihood(self, sequence):
        words = sequence.split()
        likelihoods = []
        total_possible_transitions = len(words) - self.state_size
        
        for i in range(total_possible_transitions):
            state = tuple(words[i:i+self.state_size])
            next_state = words[i+self.state_size]
            if next_state in self.transitions.get(state, {}):
                transition_count = self.transitions[state][next_state]
                likelihoods.append(transition_count)
            else:
                likelihoods.append(0)
        
        if likelihoods:
            return np.sum(likelihoods)
        else:
            return 0
        
    def weighted_geometric_mean_log_likelihood(self, sequence, alpha=0.5):
        words = sequence.split()
        valid_transitions = 0
        total_possible_transitions = len(words) - self.state_size
        transition_log_probs = []
        
        for i in range(total_possible_transitions):
            state = tuple(words[i:i+self.state_size])
            next_state = words[i+self.state_size]
            if next_state in self.transitions.get(state, {}):
                transition_count = self.transitions[state][next_state]
                state_count = self.states[state]
                if state_count > 0:
                    transition_prob = transition_count / state_count
                    transition_log_probs.append(np.log(transition_prob))
                    valid_transitions += 1
        
        if transition_log_probs:
            mean_transition_likelihood = np.exp(np.sum(transition_log_probs) / valid_transitions)
            transition_ratio = valid_transitions / total_possible_transitions
            return alpha * mean_transition_likelihood + (1 - alpha) * transition_ratio
        else:
            return 0
    
    def generate_fromprefix(self, prefixes):
        if self.state_size != 1:
            raise ValueError("This function is only for state_size=1.")
        
        prefixes = prefixes.split()  # Split the input string into individual prefixes
        generated_sequence = []
        state_length = len(list(self.states.keys())[0])

        # Generate the initial state considering the single prefix
        initial_prefix = prefixes[0] 
        possible_initial_states = [state for state in self.states.keys() if state[0].startswith(initial_prefix)]
        
        if not possible_initial_states:
            print(f"No valid initial state found with prefix '{initial_prefix}'. Using padded state '{initial_state}'.")
            initial_state = initial_prefix + 'x' * (state_length - len(initial_prefix))
            #Text to tuple
            initial_state = (initial_state,)
            generated_sequence.append(initial_state)
        else:
            initial_state = max(possible_initial_states, key=lambda state: self.states[state])
            generated_sequence.append(initial_state)

        # Generate the remaining states
        for prefix in prefixes[1:]:
            current_state = generated_sequence[-1]
            
            # Try to find the most common next state that matches the current prefix
            possible_next_states = [next_state for next_state in self.transitions[current_state] if next_state.startswith(prefix)]

            
            if possible_next_states:
                # If valid transitions exist, pick the most probable one
                next_state = max(possible_next_states, key=lambda state: self.transitions[current_state][state])
                next_state = (next_state,)
            else:
                # If no valid transitions, fallback to the most probable state with the current prefix
                possible_fallback_states = [state for state in self.states.keys() if state[0].startswith(prefix)]
                if not possible_fallback_states:
                    print(f"No valid state found with prefix '{prefix}' after state '{current_state}'. Filling with 'x'.")
                    next_state = prefix + 'x' * (state_length - len(prefix))
                    next_state = (next_state,)
                else:
                    next_state = max(possible_fallback_states, key=lambda state: self.states[state])
            
            generated_sequence.append(next_state)
        
        suffix_start_index = len(initial_prefix)
        generated_sequence = [state[0] for state in generated_sequence]
        suffix_list = [state[suffix_start_index:] for state in generated_sequence]
        return ' '.join(generated_sequence), ' '.join(suffix_list)
    
    def generate_fromsuffix(self, prefixes, suffixes):
        if self.state_size != 1:
            raise ValueError("This function is only for state_size=1.")

        prefixes = prefixes.split()
        suffixes = suffixes.split()
        if len(prefixes) != len(suffixes):
            raise ValueError("Prefixes and suffixes must be of the same length.")

        generated_sequence = []
        state_length = len(list(self.states.keys())[0])

        initial_prefix = prefixes[0]
        initial_suffix = suffixes[0]
        possible_initial_states = [state for state in self.states.keys()
                                if state[0].startswith(initial_prefix) and state[0].endswith(initial_suffix)]

        if not possible_initial_states:
            padded = initial_prefix + 'x' * (state_length - len(initial_prefix) - len(initial_suffix)) + initial_suffix
            initial_state = (padded,)
        else:
            initial_state = max(possible_initial_states, key=lambda state: self.states[state])

        generated_sequence.append(initial_state)

        for prefix, suffix in zip(prefixes[1:], suffixes[1:]):
            current_state = generated_sequence[-1]

            possible_next_states = [next_state for next_state in self.transitions[current_state]
                                    if next_state.startswith(prefix) and next_state.endswith(suffix)]

            if possible_next_states:
                next_state = max(possible_next_states, key=lambda s: self.transitions[current_state][s])
                next_state = (next_state,)
            else:
                fallback_states = [state for state in self.states.keys() if state[0].startswith(prefix) and state[0].endswith(suffix)]
                if not fallback_states:
                    padded = prefix + 'x' * (state_length - len(prefix) - len(suffix)) + suffix
                    next_state = (padded,)
                else:
                    next_state = max(fallback_states, key=lambda s: self.states[s])

            generated_sequence.append(next_state)

        generated_sequence = [state[0] for state in generated_sequence]
        return ' '.join(generated_sequence)


    def degree_distribution(self):
        return pd.Series(dict(self.graph.degree(weight='weight')))

    def clustering_coefficient(self):
        return nx.average_clustering(self.graph)

    def density(self):
        return nx.density(self.graph)

    def diameter(self):
        try:
            return nx.diameter(self.graph)
        except nx.NetworkXError:
            return float('inf')

    def size(self):
        return self.graph.number_of_nodes()

    def strong_connectivity(self):
        return nx.is_strongly_connected(self.graph)
    
    def weak_connectivity(self):
        return nx.is_weakly_connected(self.graph)


    def weights_distribution(self):
        weights = [d['weight'] for u, v, d in self.graph.edges(data=True)]
        return pd.Series(weights)

    def cycles(self):
        return list(nx.simple_cycles(self.graph))
    
    def graph_metrics(self):
        metrics = {
            'degree_distribution': self.degree_distribution(),
            'clustering_coefficient': self.clustering_coefficient(),
            'density': self.density(),
            'diameter': self.diameter(),
            'size': self.size(),
            'strong_connectivity': self.strong_connectivity(),
            'weak_connectivity': self.weak_connectivity(),
            'weights_distribution': self.weights_distribution(),
            'cycles': self.cycles()
        }
        return metrics


