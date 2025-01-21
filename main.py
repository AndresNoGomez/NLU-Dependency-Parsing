from model import ParserMLP
from utils import *

# Main function
def main():

    #Read files and get the samples or the trees
    train_samples, dev_samples, test_trees = read_files_and_preprocess_samples()

    # Initialize the ParserMLP with the best params found
    parser_mlp = ParserMLP(
        word_emb_dim = 500,
        pos_emb_dim = 500,
        hidden_dim = 400,
        epochs = 20,
        batch_size = 200,
        n_features_buff = 20,
        n_features_stack = 20,
    )

    # Build the vectorizers with the training samples
    parser_mlp.build_vectorizers_encoder(train_samples)  

    # Uncomment this line to -> Vectorize the samples and save them to the disk  
    # parser_mlp.save_vectorized_samples(train_samples, dev_samples)

    # Uncomment this line to -> Test the different params configurations
    # parser_mlp.train_models(train_samples, dev_samples, use_prevectorized=True)

    # Uncomment this line to -> Train the model and save it to the disk
    #train_save_final_model(parser_mlp, train_samples, dev_samples, "models/model_trained_final.keras",  use_prevectorized=True)

    # Load the model
    parser_mlp.load_model("models/model_trained_final.keras")         

    # Evaluate the model on the development set
    parser_mlp.evaluate(dev_samples, use_prevectorized=True)

    # Infer on the test set and evaluate the results
    infer_and_evaluate(parser_mlp, test_trees)

if __name__ == "__main__":
    main()