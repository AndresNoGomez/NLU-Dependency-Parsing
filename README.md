# NLU-Dependency-Parsing
This project implements a **transition-based dependency parsing model** that relies on an **ANN** as a scoring model to predict the best transition given a state of the **Arc-Eager** finite state machine. The training of the model can be explained as two steps: 

First, starting from an annotated **ConLLu treebank**, we extract the **samples (State, Transition) used to train the ANN**. This is done in the Algorithm.py module designing a **static Oracle** that is able to return the sequence of states and transitions that lead to the correct parsing of a sentence given the final syntactic structure that it is aiming for.

Second, to train the ANN, we transform the Arg-Eager **states to features (words and UPOS tag)** of the most notable elements of the state. The ANN consists of two separate input and embedding layers for these two features, then they are concatenated and followed by two Dense layers that allow to capture the essential information so that the model can **predict a suitable pair of Transition and Dependency Label for any given state**. Different hyperparameters and architectures were tested to ensure an optimal performance both in Transition and Label accuracies in the development ConLLu treebank.

Once the ANN is trained, the **inference of parsing trees** for unseen sentences from the test ConLLu treebank is carried out. **The F1 scores obtained were of 74.50% in Unlabeled Attachment Score (UAS) and 67.08% in Labeled Attachment Score (LAS).**

More detailed information about implementation decissions, code structure and results obtained is available on the Project Memory file.

Please note that the development of this project was limited to the scope of an academic exercise. As such, the project has not undergone further refinement or optimization beyond this point.
