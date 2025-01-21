import copy
from conllu_token import Token 

class State(object):
    """
    Class to represent a parsing state in dependency parsing.

    Attributes:
        S (list['Token']): A stack holding tokens that are 
                           currently being processed.
        B (list['Token']): A buffer holding tokens that are yet 
                           to be processed.
        A (set[tuple]): A set of arcs of the form (head_id, dependency_label, dependent_id) 
                        created during parsing, representing the dependencies.

    The class is used in dependency parsing algorithms to maintain the state of the parsing 
    process, including which tokens are being considered and the relationships formed between tokens.
    """

    def __init__(self, s: list, b: list, a: set):
        """
        Initializes a new instance of the State class.

        Parameters:
            s (list): The initial stack for the parsing state.
            b (list): The initial buffer for the parsing state.
            a (set): The initial set of arcs for the parsing state.
        """
        self._S = s
        self._B = b
        self._A = a

    @property
    def S(self):
        """
        Gets the stack of tokens currently being processed.

        Returns:
            list[Token]: The stack of tokens.
        """
        return self._S

    @property
    def B(self):
        """
        Gets the buffer of tokens yet to be processed.

        Returns:
            list[Token]: The buffer of tokens.
        """
        return self._B

    @property
    def A(self):
        """
        Gets the set of arcs created during parsing.

        Returns:
            set[tuple]: The set of arcs, each a tuple of (head_id, dependency_label, dependent_id).
        """
        return self._A


    def __str__(self):
        """
        Returns a string representation of the State instance.

        The representation includes the contents of the stack, buffer, and arcs,
        providing a snapshot of the current state of the parsing process.

        Returns:
            str: A string representation of the current parsing state.
        """        
        stack = "Stack (size={}): ".format(len(self.S)) + " | ".join([f"({e.id}, {e.form}, {e.upos})" for e in self.S]) + "\n"
        buffer = "Buffer (size={}): ".format(len(self.B)) + " | ".join([f"({e.id}, {e.form}, {e.upos})" for e in self.B]) + "\n"
        arcs = "Arcs (size={}): ".format(len(self.A)) + str(self.A) + "\n"


        return stack + buffer + arcs
    
    # Esta funcion fue añadida por nosotros para el correcto funcionamiento de arc_eager.oracle()
    def copy(self):
        """
        Creates a deep copy of the State instance.
        This ensures that changes to the original state do not affect the copied one.

        Returns:
            State: A new instance of State with the same stack, buffer, and arcs.
        """
        return State(copy.deepcopy(self._S), copy.deepcopy(self._B), copy.deepcopy(self._A))
    


if __name__ == "__main__":

    # Example tree structure (list of Token objects, already with the dummy root included as part of the tree)
    tree = [
        Token(0, "ROOT", "ROOT", "_", "_", "_", "_", "_"),
        Token(1, "The", "the", "DET", "_", "Definite=Def|PronType=Art", 2, "det"),
        Token(2, "cat", "cat", "NOUN", "_", "Number=Sing", 4, "nsubj"),
        Token(3, "is", "be", "AUX", "_", "Mood=Ind|Tense=Pres|VerbForm=Fin", 4, "cop"),
        Token(4, "sleeping", "sleep", "VERB", "_", "VerbForm=Ger", 0, "root"),
        Token(5, ".", ".", "PUNCT", "_", "_", 4, "punct")
    ]

    print ("Creating an initial state for the sentence: ")
    for token in tree:
        print (token)

    state = State(s=[tree[0]], b=tree[1:], a=set([]))
    print(state)

    print ("Creating a random state and accessing to some positions")

    #Example of random state
    random_state = State(s=[tree[0],tree[2]], b=tree[3:], a=set([(2, "det", 1)]))

    print()
    # Displaying the complete state
    print("Whole Random State:")
    print(random_state)

    # Accessing elements from the stack
    top_of_stack = random_state.S[-1]
    print("Top of Stack Token:", top_of_stack)
    second_top_of_stack =  random_state.S[-2]
    print ("Second top of Stack Token: ", second_top_of_stack)
    print()
    # Accessing elements from the buffer
    first_in_buffer = random_state.B[0]
    print("First in Buffer:", first_in_buffer)
    second_in_buffer = random_state.B[1]
    print("Second in Buffer:", second_in_buffer)
    print()
    # Understanding the format of the arcs attribute
    # The 'A' attribute is a set of tuples, each tuple representing an arc
    # Format of each tuple in the arcs set: (head_id, dependency_label, dependent_id)
    for arc in random_state.A:
        print("Arc:", arc)