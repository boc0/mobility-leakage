import pickle
from markov_model import MarkovModel

def prepare_corpus_from_deepmove(data_pickle_path):
    """
    Loads the processed data from sparse_traces.py and converts it
    into a corpus for the MarkovModel.
    """
    print(f"Loading data from {data_pickle_path}...")
    with open(data_pickle_path, 'rb') as f:
        data = pickle.load(f)

    data_neural = data['data_neural']
    corpus = []

    # Iterate through each user and their sessions
    for user_id in data_neural:
        sessions = data_neural[user_id]['sessions']
        for session_id in sessions:
            session = sessions[session_id]
            
            # Extract only the location IDs (vid) from the session
            # A session looks like: [[vid1, tid1], [vid2, tid2], ...]
            location_ids = [str(point[0]) for point in session]
            
            # A Markov "sentence" must have at least state_size + 1 words
            # If your state_size is 1, you need at least 2 locations.
            if len(location_ids) > 1:
                sentence = ' '.join(location_ids)
                corpus.append(sentence)
                
    print(f"Created a corpus with {len(corpus)} sequences.")
    return corpus

if __name__ == '__main__':
    # 1. Define the path to the output file from sparse_traces.py
    processed_data_path = 'DeepMove/data/foursquare.pk' # Or whatever you named it

    # 2. Prepare the corpus
    trajectory_corpus = prepare_corpus_from_deepmove(processed_data_path)

    # 3. Initialize and train the Markov Model
    # state_size=1 means it predicts the next location based on the last one.
    # state_size=2 would use the last two locations.
    markov = MarkovModel(state_size=1)
    print("Training Markov model...")
    markov.train(trajectory_corpus)
    print("Training complete.")

    # 4. Save the trained model
    model_save_path = 'markov_model.json'
    markov.save_json(model_save_path)

    # 5. Load the model from the file
    loaded_markov = MarkovModel.load_json(model_save_path)

    # Now the loaded model is ready to be used
    if loaded_markov.states:
        generated_sequence = loaded_markov.generate(length=4)
        print(f"Generated sequence from loaded model: {generated_sequence}")
        likelihood = loaded_markov.likelihood(generated_sequence)
        print(f"Likelihood of generated sequence: {likelihood}")
    else:
        print("No states were learned, cannot generate a sequence.")