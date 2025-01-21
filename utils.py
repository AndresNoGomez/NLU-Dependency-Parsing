from conllu_reader import ConlluReader
from algorithm import ArcEager
from postprocessor import PostProcessor
import os

# Read the conllu files
def read_file(reader, path, inference):
    trees = reader.read_conllu_file(path, inference)
    print(f"Read a total of {len(trees)} sentences from {path}")
    return trees


#Create conllu file with the trees from the run
def postprocess_trees_and_save_to_conllu_file(parsed_trees, output_path):
    reader = ConlluReader()
    post_processor = PostProcessor()    

    # Write the parsed trees to a temporary CoNLLU file
    temp_file = "temp_parsed_trees.conllu"
    reader.write_conllu_file(temp_file, parsed_trees)

    # Postprocess and write the temporary file
    processed_trees = post_processor.postprocess(temp_file)
    reader.write_conllu_file(output_path, processed_trees)

    # Remove the temporary file
    os.remove(temp_file)
    print(f"Parsed trees have been saved to {output_path} in CoNLLU format.")


# Read the conllu files and remove non-projective sentences
def read_files_and_preprocess_samples():
    # Initialize the ConlluReader
    reader = ConlluReader()
    train_trees = read_file(reader,path="datasets/en_partut-ud-train_clean.conllu", inference=False)
    dev_trees = read_file(reader,path="datasets/en_partut-ud-dev_clean.conllu", inference=False)
    test_trees = read_file(reader,path="datasets/en_partut-ud-test_clean.conllu", inference=True)

    # Remove non-projective sentences from the training and development sets
    train_trees = reader.remove_non_projective_trees(train_trees)
    dev_trees = reader.remove_non_projective_trees(dev_trees)

    print ("Total training trees after removing non-projective sentences", len(train_trees))
    print ("Total dev trees after removing non-projective sentences", len(dev_trees))
    
    # Generate training and development samples using the ArcEager algorithm
    arc_eager = ArcEager()
    train_samples = [sample for sent in train_trees for sample in arc_eager.oracle(sent)]
    dev_samples = [sample for sent in dev_trees for sample in arc_eager.oracle(sent)]

    return train_samples, dev_samples, test_trees


# Train the model and save the final model
def train_save_final_model(parser_mlp, training_samples, dev_samples, model_path, use_prevectorized=False):
    # Train the parser model
    parser_mlp.train(training_samples, dev_samples, use_prevectorized)

    # Save the model
    parser_mlp.save_model(model_path)


def infer_and_evaluate(parser_mlp, test_trees):
    # Infer on the test set
    predicted_test_trees = parser_mlp.run(test_trees) 

    # Save the predicted trees to a CoNLLU file
    output_file_path_test = "results/parsed_test_set.conllu"
    postprocess_trees_and_save_to_conllu_file(predicted_test_trees, output_file_path_test)

    # Evaluate the predicted trees
    os.system("python ./conll18_ud_eval.py ./datasets/en_partut-ud-test_clean.conllu ./results/parsed_test_set.conllu -v")