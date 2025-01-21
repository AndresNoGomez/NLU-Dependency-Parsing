from conllu_token import Token
from algorithm import Sample, Transition, ArcEager
from keras import models, layers, callbacks
import numpy as np
from sklearn.preprocessing import LabelEncoder

class ParserMLP:
    """
    A Multi-Layer Perceptron (MLP) class for a dependency parser, using TensorFlow and Keras.

    This class implements a neural network model designed to predict transitions in a dependency 
    parser. It utilizes the Keras Functional API, which is more suited for multi-task learning scenarios 
    like this one. The network is trained to map parsing states to transition actions, facilitating 
    the parsing process in natural language processing tasks.

    Attributes:
        word_emb_dim (int): Dimensionality of the word embeddings. Defaults to 100.
        hidden_dim (int): Dimension of the hidden layer in the neural network. Defaults to 64.
        epochs (int): Number of training epochs. Defaults to 1.
        batch_size (int): Size of the batches used in training. Defaults to 64.

    Methods:
        train(training_samples, dev_samples): Trains the MLP model using the provided training and 
            development samples. It maps these samples to IDs that can be processed by an embedding 
            layer and then calls the Keras compile and fit functions.

        evaluate(samples): Evaluates the performance of the model on a given set of samples. The 
            method aims to assess the accuracy in predicting both the transition and dependency types, 
            with expected accuracies ranging between 75% and 85%.

        run(sents): Processes a list of sentences (tokens) using the trained model to perform dependency 
            parsing. This method implements the vertical processing of sentences to predict parser 
            transitions for each token.

        Feel free to add other parameters and functions you might need to create your model
    """

    def __init__(self, word_emb_dim: int = 500, pos_emb_dim: int = 500, hidden_dim: int = 400, 
                 epochs: int = 20, batch_size: int = 200, n_features_buff: int = 20, n_features_stack: int = 20):
        """
        Initializes the ParserMLP class.

        Parameters:
            word_emb_dim (int): Dimensionality of word embeddings.
            pos_emb_dim (int): Dimensionality of UPOS embeddings.
            hidden_dim (int): Number of units in the hidden layer.
            max_length (int): Maximum length of the input features (stack + buffer size).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            n_features_buff (int): Number of features for the buffer.
            n_features_stack (int): Number of features for the stack.
        """
        self.word_emb_dim = word_emb_dim
        self.pos_emb_dim = pos_emb_dim 
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_features_buff = n_features_buff
        self.n_features_stack = n_features_stack
        self.n_features = self.n_features_buff + self.n_features_stack

        # Text vectorizers, encoder and mapping
        self.word_vectorizer = layers.TextVectorization(output_mode="int", output_sequence_length=self.n_features)
        self.upos_vectorizer = layers.TextVectorization(output_mode="int", output_sequence_length=self.n_features)
        self.dependency_encoder = LabelEncoder()
        self.ACTION_MAP = {
        "SHIFT": 0,
        "LEFT-ARC": 1,
        "RIGHT-ARC": 2,
        "REDUCE": 3
        }
        
        self.model = None


    def build_model(self):
        """
        Builds the Model to be used in the ParserMLP class.
        """
        print("\nBuilding the model...", end='\r')
        # Input word layer
        word_input = layers.Input(shape=(self.n_features,), name="word_input", dtype="int32")
        word_embedding = layers.Embedding(
            input_dim=self.word_vectorizer.vocabulary_size(),
            output_dim=self.word_emb_dim,
            name="word_embedding"
        )(word_input)

        # Input UPOS layer
        upos_input = layers.Input(shape=(self.n_features,), name="upos_input", dtype="int32")
        upos_embedding = layers.Embedding(
            input_dim=self.upos_vectorizer.vocabulary_size(),
            output_dim=self.pos_emb_dim,
            name="upos_embedding"
        )(upos_input)

        #Flatten the embeddings
        word_flatten = layers.Flatten(name="word_flatten")(word_embedding)
        upos_flatten = layers.Flatten(name="upos_flatten")(upos_embedding)

        # Concatenate the flattens
        combined_f = layers.Concatenate(name="combine_embeddings")([word_flatten, upos_flatten])
        
        # Hidden layer
        hidden_1 = layers.Dense(self.hidden_dim, activation='relu', name="hidden_layer_1")(combined_f)
        hidden_2 = layers.Dense(self.hidden_dim, activation='relu', name="hidden_layer_2")(combined_f)

        # Output layer for transitions
        transition_output = layers.Dense(4, activation='softmax', name="transition_output")(hidden_1)

        # Output layer for dependency labels
        dependency_output = layers.Dense(len(self.dependency_encoder.classes_), activation='softmax', name="dependency_output")(hidden_2)
        
        # Define the model with specified inputs and outputs
        self.model = models.Model(inputs=(word_input, upos_input), outputs=(transition_output, dependency_output))

        # Compile the model with multi-task loss and metrics
        self.model.compile(
            optimizer='adam',
            loss={
                "transition_output": "sparse_categorical_crossentropy",
                "dependency_output": "sparse_categorical_crossentropy",
            },
            metrics={
                "transition_output": ['accuracy'],
                "dependency_output": ['accuracy']
            }
        )

        print("Model built successfully!")
        self.model.summary()



    def load_model(self, model_path: str):
        """
        Loads a pre-trained model from the specified file path.

        Parameters:
            model_path (str): The file path to load the model from.
        """
        print(f"\nLoading model from {model_path}...", end='\r')
        self.model = models.load_model(model_path)
        print(f"Model loaded from {model_path} successfully!")



    def save_model(self, model_path: str):
        """
        Saves the current model to the specified file path.

        Parameters:
            model_path (str): The file path to save the model to.
        """
        print(f"\nSaving model to {model_path}...", end='\r')
        self.model.save(model_path)
        print(f"Model saved to {model_path} successfully!")



    def build_vectorizers_encoder(self, training_samples: list['Sample']):
        """
        Builds and fits the vectorizers and encoder using the provided training samples.

        Parameters:
            training_samples (list[Sample]): A list of training samples for the parser.
        """
        print("\nBuilding vectorizers and encoder...", end='\r')
        # Extract features from the training samples
        train_input = [sample.state_to_feats(nbuffer_feats=self.n_features_buff, nstack_feats=self.n_features_stack) for sample in training_samples]
        train_words, train_upos = [features[:len(features) // 2] for features in train_input], [features[len(features) // 2:] for features in train_input]  

        # Extract targets for fitting the dependency encoder
        train_targets_dependency = [sample.transition.dependency if sample.transition.dependency is not None else "None" for sample in training_samples]

        # Fit the vectorizers and encoder
        self.word_vectorizer.adapt([" ".join(sequence) for sequence in train_words])
        self.upos_vectorizer.adapt([" ".join(sequence) for sequence in train_upos])
        self.dependency_encoder.fit(train_targets_dependency)

        print("Vectorizers and encoder built successfully!")



    def vectorize_samples(self, samples: list['Sample']):
        """
        Vectorizes the provided samples into word vectors, UPOS vectors, and target labels.

        Parameters:
            samples (list[Sample]): A list of samples to be vectorized.

        Returns:
            tuple: (word_vectors, upos_vectors, action_labels, dependency_labels)
        """
        # Extract features from the samples
        input_data = [sample.state_to_feats(nbuffer_feats=self.n_features_buff, nstack_feats=self.n_features_stack) for sample in samples]
        words, upos = [features[:len(features) // 2] for features in input_data], [features[len(features) // 2:] for features in input_data]
        
        # Vectorize the words and UPOS tags
        word_vectors = np.array([self.word_vectorizer(" ".join(seq)).numpy() for seq in words], dtype=np.int32)
        upos_vectors = np.array([self.upos_vectorizer(" ".join(seq)).numpy() for seq in upos], dtype=np.int32)

        # Vectorize the target actions and dependencies, if the transition is not None.
        action_labels = np.array([self.ACTION_MAP[sample.transition.action] for sample in samples if sample.transition is not None], dtype=np.int32)
        dependency_labels = np.array(
            [self.dependency_encoder.transform([sample.transition.dependency if sample.transition.dependency is not None else "None"])[0] for sample in samples if sample.transition is not None],
            dtype=np.int32
        )
        return word_vectors, upos_vectors, action_labels, dependency_labels



    def save_vectorized_samples(self, training_samples: list['Sample'], dev_samples: list['Sample']):
        # Vectorize the training and development samples
        print("\nVectorizing and saving samples...", end='\r')
        train_word_vectors, train_upos_vectors, train_targets_actions, train_targets_dependencies = self.vectorize_samples(training_samples)
        dev_word_vectors, dev_upos_vectors, dev_targets_actions, dev_targets_dependencies = self.vectorize_samples(dev_samples)

        # Save the vectorized samples
        np.save('datasets/vectorized_samples/train_word_vectors.npy', train_word_vectors)
        np.save('datasets/vectorized_samples/train_upos_vectors.npy', train_upos_vectors)
        np.save('datasets/vectorized_samples/train_targets_actions.npy', train_targets_actions)
        np.save('datasets/vectorized_samples/train_targets_dependencies.npy', train_targets_dependencies)
        np.save('datasets/vectorized_samples/dev_word_vectors.npy', dev_word_vectors)
        np.save('datasets/vectorized_samples/dev_upos_vectors.npy', dev_upos_vectors)
        np.save('datasets/vectorized_samples/dev_targets_actions.npy', dev_targets_actions)
        np.save('datasets/vectorized_samples/dev_targets_dependencies.npy', dev_targets_dependencies)
        
        print("Samples vectorized and saved successfully!")

    def load_vectorized_samples(self):
        print("\nLoading pre-vectorized samples...", end='\r')

        # Load pre-vectorized samples from files
        train_word_vectors = np.load('datasets/vectorized_samples/train_word_vectors.npy')
        train_upos_vectors = np.load('datasets/vectorized_samples/train_upos_vectors.npy')
        train_targets_actions = np.load('datasets/vectorized_samples/train_targets_actions.npy')
        train_targets_dependencies = np.load('datasets/vectorized_samples/train_targets_dependencies.npy')
        dev_word_vectors = np.load('datasets/vectorized_samples/dev_word_vectors.npy')
        dev_upos_vectors = np.load('datasets/vectorized_samples/dev_upos_vectors.npy')
        dev_targets_actions = np.load('datasets/vectorized_samples/dev_targets_actions.npy')
        dev_targets_dependencies = np.load('datasets/vectorized_samples/dev_targets_dependencies.npy')

        print("Pre-vectorized samples loaded successfully!")
        return train_word_vectors, train_upos_vectors, train_targets_actions, train_targets_dependencies, dev_word_vectors, dev_upos_vectors, dev_targets_actions, dev_targets_dependencies




    def train(self, training_samples: list['Sample'], dev_samples: list['Sample'], use_prevectorized: bool = False):
        """
        Trains the MLP model using the provided training and development samples.

        This method prepares the training data by mapping samples to IDs suitable for 
        embedding layers and then proceeds to compile and fit the Keras model.

        Parameters:
            training_samples (list[Sample]): A list of training samples for the parser.
            dev_samples (list[Sample]): A list of development samples used for model validation.
            use_prevectorized (bool): A flag indicating whether to use pre-vectorized samples, defaults to False.
        """

        if use_prevectorized:
            train_word_vectors, train_upos_vectors, train_targets_actions, train_targets_dependencies, dev_word_vectors, dev_upos_vectors, dev_targets_actions, dev_targets_dependencies = self.load_vectorized_samples()
        else:
            # Vectorize the training and development samples
            print("\nVectorizing samples...", end='\r')
            train_word_vectors, train_upos_vectors, train_targets_actions, train_targets_dependencies = self.vectorize_samples(training_samples)
            dev_word_vectors, dev_upos_vectors, dev_targets_actions, dev_targets_dependencies = self.vectorize_samples(dev_samples)
            print("Samples vectorized successfully!")

        # Build the model
        self.build_model()
        
        print("\nTraining the model...", end='\r')
        # Train the model
        self.model.fit(
            [train_word_vectors, train_upos_vectors],
            [train_targets_actions, train_targets_dependencies],  
            validation_data=([dev_word_vectors, dev_upos_vectors], [dev_targets_actions, dev_targets_dependencies]),  
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[callbacks.EarlyStopping(monitor='val_dependency_output_accuracy', patience=4, restore_best_weights=True, mode='max')],
            verbose=0
        )
        print("Model trained successfully!")


    def evaluate(self, samples: list['Sample'], use_prevectorized: bool = False):
        """
        Evaluates the model's performance on a set of samples.

        This method is used to assess the accuracy of the model in predicting the correct
        transition and dependency types. The expected accuracy range is between 75% and 85%.

        Parameters:
            samples (list[Sample]): A list of samples to evaluate the model's performance.
            use_prevectorized (bool): A flag indicating whether to use pre-vectorized samples, defaults to False.
            
        Returns:
        metrics (dict): A dictionary containing loss and accuracy metrics for transitions and dependencies.
        """
        print("\nEvaluating the model on the development set...", end='\r')

        # Vectorize the test samples
        if use_prevectorized:
            _, _, _, _, eval_word_vectors, eval_upos_vectors, eval_targets_actions, eval_targets_dependencies = self.load_vectorized_samples()
        else:
            # Vectorize the evaluation samples
            eval_word_vectors, eval_upos_vectors, eval_targets_actions, eval_targets_dependencies = self.vectorize_samples(samples)

        # Evaluate the model
        results = self.model.evaluate(
            [eval_word_vectors, eval_upos_vectors],
            [eval_targets_actions, eval_targets_dependencies],
            batch_size=self.batch_size,
            verbose=0
        )

        # Map the results to specific metrics
        metrics = {
        "total_loss": results[0],  
        "transition_loss": results[1],
        "dependency_loss": results[2],
        "transition_accuracy": results[3],
        "dependency_accuracy": results[4] 
        }

        print("Evaluation metrics: ", metrics)
        return metrics



    def train_models(self, training_samples: list['Sample'], dev_samples: list['Sample'], use_prevectorized: bool = False):
        """
        Trains multiple models with different hyperparameters 

        Parameters:
            training_samples (list[Sample]): A list of training samples for the parser.
            dev_samples (list[Sample]): A list of development samples used for model validation.
            use_prevectorized (bool): A flag indicating whether to use pre-vectorized samples, defaults to False.
        """
        params = [
        #First observations
            {"word_emb_dim": 50, "pos_emb_dim": 50, "hidden_dim": 50, "epochs": 15, "batch_size": 50, "n_features_buff": 5, "n_features_stack": 5},
            {"word_emb_dim": 100, "pos_emb_dim": 100, "hidden_dim": 100, "epochs": 15, "batch_size": 50, "n_features_buff": 8, "n_features_stack": 8},
            {"word_emb_dim": 200, "pos_emb_dim": 200, "hidden_dim": 200, "epochs": 15, "batch_size": 100, "n_features_buff": 12, "n_features_stack": 12},
            {"word_emb_dim": 300, "pos_emb_dim": 300, "hidden_dim": 300, "epochs": 20, "batch_size": 100, "n_features_buff": 16, "n_features_stack": 16},
            {"word_emb_dim": 400, "pos_emb_dim": 400, "hidden_dim": 400, "epochs": 20, "batch_size": 200, "n_features_buff": 20, "n_features_stack": 20},

        #Then, we can try with different hidden dimensions
            {"word_emb_dim": 400, "pos_emb_dim": 400, "hidden_dim": 50, "epochs": 20, "batch_size": 200, "n_features_buff": 20, "n_features_stack": 20},
            {"word_emb_dim": 400, "pos_emb_dim": 400, "hidden_dim": 100, "epochs": 20, "batch_size": 200, "n_features_buff": 20, "n_features_stack": 20},
            {"word_emb_dim": 400, "pos_emb_dim": 400, "hidden_dim": 200, "epochs": 20, "batch_size": 200, "n_features_buff": 20, "n_features_stack": 20},
            {"word_emb_dim": 400, "pos_emb_dim": 400, "hidden_dim": 300, "epochs": 20, "batch_size": 200, "n_features_buff": 20, "n_features_stack": 20},
            {"word_emb_dim": 400, "pos_emb_dim": 400, "hidden_dim": 400, "epochs": 20, "batch_size": 200, "n_features_buff": 20, "n_features_stack": 20},
        
        # Then, we can try with more epochs
            {"word_emb_dim": 50, "pos_emb_dim": 50, "hidden_dim": 400, "epochs": 20, "batch_size": 200, "n_features_buff": 20, "n_features_stack": 20},
            {"word_emb_dim": 100, "pos_emb_dim": 100, "hidden_dim": 400, "epochs": 20, "batch_size": 200, "n_features_buff": 20, "n_features_stack": 20},
            {"word_emb_dim": 200, "pos_emb_dim": 200, "hidden_dim": 400, "epochs": 20, "batch_size": 200, "n_features_buff": 20, "n_features_stack": 20},
            {"word_emb_dim": 300, "pos_emb_dim": 300, "hidden_dim": 400, "epochs": 20, "batch_size": 200, "n_features_buff": 20, "n_features_stack": 20},
            {"word_emb_dim": 500, "pos_emb_dim": 500, "hidden_dim": 400, "epochs": 20, "batch_size": 200, "n_features_buff": 20, "n_features_stack": 20},
            
        # Now we can try with more features
            {"word_emb_dim": 400, "pos_emb_dim": 400, "hidden_dim": 400, "epochs": 20, "batch_size": 200, "n_features_buff": 8, "n_features_stack": 8},
            {"word_emb_dim": 400, "pos_emb_dim": 400, "hidden_dim": 400, "epochs": 20, "batch_size": 200, "n_features_buff": 12, "n_features_stack": 12},
            {"word_emb_dim": 400, "pos_emb_dim": 400, "hidden_dim": 400, "epochs": 20, "batch_size": 200, "n_features_buff": 16, "n_features_stack": 16},
            {"word_emb_dim": 400, "pos_emb_dim": 400, "hidden_dim": 400, "epochs": 20, "batch_size": 200, "n_features_buff": 24, "n_features_stack": 24},
            {"word_emb_dim": 400, "pos_emb_dim": 400, "hidden_dim": 400, "epochs": 20, "batch_size": 200, "n_features_buff": 28, "n_features_stack": 28},
        ]
        
        # Vectorize the training and development samples
        if use_prevectorized:
            train_word_vectors, train_upos_vectors, train_targets_actions, train_targets_dependencies, dev_word_vectors, dev_upos_vectors, dev_targets_actions, dev_targets_dependencies = self.load_vectorized_samples()
        else:
            # Vectorize the training and development samples
            print("\nVectorizing samples...", end='\r')
            train_word_vectors, train_upos_vectors, train_targets_actions, train_targets_dependencies = self.vectorize_samples(training_samples)
            dev_word_vectors, dev_upos_vectors, dev_targets_actions, dev_targets_dependencies = self.vectorize_samples(dev_samples)
            print("Samples vectorized successfully!")

        # Train the models
        for i, param in enumerate(params):
            print(f"\nTraining model {i + 1}...", end='\r')
            
            # Build the model with the params or the default values
            self.word_emb_dim = param.get("word_emb_dim", 100)
            self.pos_emb_dim = param.get("pos_emb_dim", 100)
            self.hidden_dim = param.get("hidden_dim", 100)
            self.epochs = param.get("epochs", 10)
            self.batch_size = param.get("batch_size", 64)
            self.build_model()

            # Train the model
            self.model.fit(
                [train_word_vectors, train_upos_vectors],
                [train_targets_actions, train_targets_dependencies],  
                validation_data=([dev_word_vectors, dev_upos_vectors], [dev_targets_actions, dev_targets_dependencies]),  
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[callbacks.EarlyStopping(monitor='val_dependency_output_accuracy', patience=4, restore_best_weights=True, mode='max')],
                verbose=0
            )

            print(f"Model {i + 1} trained successfully!")

            # Evaluate the model trained
            self.evaluate(dev_samples)



    
    def run(self, sents: list['Token']) -> list[list['Token']]:
        """
        Executes the model on a list of sentences to perform dependency parsing.

        Parameters:
            sents (list[Token]): A list of sentences, where each sentence is represented 
                                as a list of Token objects.

        Returns:
            list[list[Token]]: A list of parsed trees (each tree is a list of Token objects).
        """
        print("\nRunning the model on the test set...", end='\r')

        # Initialize 
        arc_eager = ArcEager()
        final_states_sents = []  
        remaining_sents = list(enumerate(sents))
        batch_size = 100
        parsed_trees_with_indices = []

        while remaining_sents:
            # Step 1: Take a batch of input sentences
            batch_sents = remaining_sents[:batch_size]
            remaining_sents = remaining_sents[batch_size:]

            # Step 2: Obtain the corresponding batch of initial states
            batch_states_sents = [(arc_eager.create_initial_state(sent), idx, sent) for idx, sent in batch_sents]

            while batch_states_sents:
                # Step 3: Vectorize the features so that they can be used as input to the model
                batch_samples = [Sample(state, None) for state, _, _ in batch_states_sents]
                words_vectorized, upos_vectorized, _, _ = self.vectorize_samples(batch_samples)

                # Step 4: Make predictions with the trained model at the batch level.
                batch_predictions = self.model.predict([words_vectorized, upos_vectorized], verbose=0)
                batch_action_probabilities, batch_deprel_probabilities = batch_predictions[0], batch_predictions[1]

                # Step 5: Update the states using the predicted transitions 
                updated_states_sents = []

                # Sort the transition actions by probability
                batch_sorted_action_indices = np.argsort(-batch_action_probabilities, axis=1)
                index_to_action = {v: k for k, v in self.ACTION_MAP.items()}
                sorted_actions = [[index_to_action[idx] for idx in sorted_action_indices]
                                for sorted_action_indices in batch_sorted_action_indices]

                # Sort the dependency predictions by probability
                batch_sorted_dependency_indices = np.argsort(-batch_deprel_probabilities, axis=1)
                index_to_dependency = {i: dep for i, dep in enumerate(self.dependency_encoder.classes_)}
                sorted_dependencies = [[index_to_dependency[idx] for idx in sorted_dep_indices]
                                        for sorted_dep_indices in batch_sorted_dependency_indices]

                for i, (state, idx, initial_sent) in enumerate(batch_states_sents):
                    # Get the best action that satisfies the preconditions
                    selected_transition = None

                    # Try actions in order of probability
                    for action in sorted_actions[i]:
                        # Create a transition with the action and the most probable dependency
                        transition = Transition(action, sorted_dependencies[i][0])
                        # Check if the transition is valid
                        if (
                            (action == ArcEager.SHIFT) or
                            (action == ArcEager.LA and arc_eager.LA_is_valid(state)) or
                            (action == ArcEager.RA and arc_eager.RA_is_valid(state)) or
                            (action == ArcEager.REDUCE and arc_eager.REDUCE_is_valid(state))
                        ):
                            # If the action is LA or RA and the dependency is "None", select the second most probable dependency
                            if (action == ArcEager.LA or action == ArcEager.RA) and sorted_dependencies[i][0] == "None":
                                selected_transition = Transition(action, sorted_dependencies[i][1])
                            else:
                                selected_transition = transition
                            break

                    # Update the state with the selected transition
                    arc_eager.apply_transition(state, selected_transition)

                    # Add the updated states to the lists of updated and final states
                    if not arc_eager.final_state(state):
                        updated_states_sents.append((state, idx, initial_sent))
                    else:
                        final_states_sents.append((state, idx, initial_sent))

                # Update the batch states with the updated states
                batch_states_sents = updated_states_sents

            
        # Convert the final states into parsed trees
        for state, idx, initial_sent in final_states_sents:
            for arc in state.A:
                head_id, dep, dep_id = arc
                initial_sent[dep_id].head = head_id
                initial_sent[dep_id].dep = dep

            parsed_trees_with_indices.append((idx, initial_sent))

        # Reorder the parsed trees by the original indices
        parsed_trees_with_indices.sort(key=lambda x: x[0])

        # Remove indices and return only the parsed trees
        parsed_trees = [tree for _, tree in parsed_trees_with_indices]
        return parsed_trees

        
if __name__ == "__main__":
    model = ParserMLP()