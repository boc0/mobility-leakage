import argparse
import os
import pickle
from model import MarkovModel
import pandas as pd

def compute_likelihoods_for_file(model, pk_file_path, output_csv_path, likelihood_func_name):
    """
    Loads a .pk file, computes likelihoods for all its sequences using the specified
    function, and saves the results to a CSV file.
    """
    print(f"Processing {pk_file_path}...")
    try:
        with open(pk_file_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"  Error: Could not find file {pk_file_path}")
        return

    # Get the requested likelihood function from the model instance
    try:
        likelihood_function = getattr(model, 'log_likelihood')
    except AttributeError:
        print(f"  Error: Likelihood function '{likelihood_func_name}' not found in MarkovModel class.")
        return

    data_neural = data.get('data_neural', {})
    # Create a reverse map from internal user index back to original trajectory ID (tid)
    user_map = {v[0]: k for k, v in data.get('uid_list', {}).items()}
    
    results = []
    for user_idx, udata in data_neural.items():
        tid = user_map.get(user_idx, 'unknown_user')
        for sess_id, sess in udata.get('sessions', {}).items():
            # A sequence must be longer than the model's state size to have at least one transition
            if len(sess) <= model.state_size:
                continue
            
            sequence_str = ' '.join([str(p[0]) for p in sess])
            
            # Compute the likelihood using the chosen function
            likelihood = likelihood_function(sequence_str)
            
            results.append({
                'tid': tid,
                'session_id': sess_id,
                'likelihood': likelihood,
                'sequence': sequence_str
            })

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False)
        print(f"  Saved {len(results)} likelihoods to {output_csv_path}")
    else:
        print(f"  No valid sequences found in {pk_file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute trajectory likelihoods using a pre-trained Markov model."
    )
    parser.add_argument('--model', type=str, required=True,
                        help="Path to the trained markov_model.json file.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Path to the directory containing processed .pk data files.")
    parser.add_argument('--output_dir', type=str, default='markov_likelihoods',
                        help="Directory to save the output .csv files.")
    parser.add_argument('--likelihood_func', type=str, required=True,
                        choices=[
                            'likelihood', 'likelihood_with_smoothing', 'log_likelihood', 'geometric_mean_likelihood',
                            'max_normalized_transition_likelihood', 'powered_max_norm_likelihood',
                            'weighted_double_mean_likelihood', 'state_normalized_likelihood'
                        ],
                        help="The name of the likelihood function to use from the MarkovModel class.")
    
    args = parser.parse_args()

    # --- Load Model ---
    print(f"Loading Markov model from {args.model}...")
    model = MarkovModel.load_json(args.model)

    # --- Prepare Directories ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    pk_files = [f for f in os.listdir(args.data_dir) if f.endswith('.pk')]
    if not pk_files:
        print(f"No .pk files found in directory: {args.data_dir}")
        return

    # --- Main Processing Loop ---
    for pk_file in pk_files:
        input_path = os.path.join(args.data_dir, pk_file)
        output_filename = os.path.splitext(pk_file)[0] + f'_{args.likelihood_func}.csv'
        output_path = os.path.join(args.output_dir, output_filename)
        
        compute_likelihoods_for_file(model, input_path, output_path, args.likelihood_func)

    print("\nLikelihood computation complete.")


if __name__ == '__main__':
    main()