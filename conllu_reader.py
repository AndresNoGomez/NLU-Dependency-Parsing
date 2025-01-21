from conllu_token import Token 
import sys

class ConlluReader():
    """
    A class for reading and processing CoNLLU format files. CoNLLU (Conference on Natural Language 
    Learning Universal Dependencies) format is widely used for linguistic annotation, especially in 
    dependency parsing. This class provides methods to convert between tree structures (as a list of 
    Token objects) and CoNLLU formatted strings, as well as to read CoNLLU files.

    Methods:
        tree2conllustr(tree: list[Token]) -> str:
            Converts a tree structure (list of Token objects) into a CoNLLU formatted string.

        conllustr2tree(conllustr: str, inference: bool = True) -> list[Token]:
            Converts a CoNLLU formatted string into a list of Token objects representing a tree.

        read_conllu_file(path: str, inference: bool = False) -> list:
            Reads a CoNLLU file and converts it into a list of tree structures.

        write_conllu_file(path: str, trees: list['Token]):
            Writes a list of tree structures to a file in CoNLLU format.

        remove_non_projective_trees(trees: list['Token']) -> list:
            Filters out non-projective trees from a list of tree structures.

    Note: Example usage is provided at the end of the file.
    """

    def tree2conllustr(self, tree: list['Token']) -> str:
        """
        Converts a tree structure into a CoNLLU formatted string.

        Parameters:
            tree (list['Token']): A list of Token objects representing a sentence.

        Returns:
            str: A string in CoNLLU format representing the sentence.
        """
        tree_str = []
        for entry in tree:
            if entry.id == 0:
                continue
            tree_str.append("\t".join(map(str, entry.get_fields_list())))
        return "\n".join(tree_str)


    def conllustr2tree(self, conllustr: str, inference=True) -> list['Token']:
        """
        Converts a CoNLLU formatted string into a list of Token objects that represents the tree

        Parameters:
            conllustr (str): A string representing the sentence in CoNLLU format.

            inference (bool): If True, the head and dependency relation (dep) for each token 
                          are not assigned during parsing. This is useful when these 
                          attributes are to be inferred or predicted later. If False, 
                          these attributes are assigned based on the CoNLLU input.

        Returns:
            list[Token]: A list of Token objects representing the parsed tree.
        """
        lines = conllustr.strip().split("\n")
        dummy_root = Token(0, "ROOT", "ROOT", "ROOT_UPOS", "ROOT_CPOS", "ROOT_FEATS", "_", "_")
        tree = [dummy_root]

        for line in lines:
            if not (self._line_is_comment(line) or 
                    self._line_is_multiword(line) or 
                    self._line_is_empty_token(line)):
                columns = line.split("\t")
                if len(columns) >= 8:
                    try:
                        #We do not assign the head and deprel columns if we are going to predict them
                        if inference:
                            token = Token( int(columns[0]), columns[1], columns[2], columns[3],
                            columns[4], columns[5], "_", "_")                            
                        else:
                            token = Token(int(columns[0]), columns[1], columns[2], columns[3],
                            columns[4], columns[5], int(columns[6]), columns[7]
                        )
                        tree.append(token)
                    except ValueError:
                        #We'll need it to read with this function corrupted file produced by the model
                        #to later postprocess them with the Postprocessor class.
                        token = Token(int(columns[0]), columns[1], columns[2], columns[3],
                            columns[4], columns[5], "_", columns[7])
                        tree.append(token)

        return tree


    def _line_is_comment(self, line: str) -> bool:
        """
        Checks if a line in a CoNLLU formatted string is a comment.

        Parameters:
            line (str): A line from a CoNLLU formatted string.

        Returns:
            bool: True if the line is a comment, False otherwise.
        """
        return line.startswith("#")

    def _line_is_multiword(self, line: str) -> bool:
        """
        Checks if a line in a CoNLLU formatted string represents a multiword token.

        Parameters:
            line (str): A line from a CoNLLU formatted string.

        Returns:
            bool: True if the line represents a multiword token, False otherwise.
        """
        id_column = line.split("\t")[0]
        return "-" in id_column

    def _line_is_empty_token(self, line: str) -> bool:
        """
        Checks if a line in a CoNLLU formatted string represents an empty token.

        Parameters:
            line (str): A line from a CoNLLU formatted string.

        Returns:
            bool: True if the line represents an empty token, False otherwise.
        """
        id_column = line.split("\t")[0]
        return '.' in id_column
    
    def _is_projective(self,tree: list['Token']) -> bool:
        """
            Determines if a dependency tree has crossing arcs or not.

            Parameters:
                tree (list['Token']): A list of Token instances representing a dependency tree

            Returns:
                A boolean: True if the tree is projective, False otherwise
        """
        arcs = [(entry.head, entry.id) for entry in tree if entry.id !=0]
        for (i,j) in arcs:
            for (k,l) in arcs:
                if (i,j) != (k,l) and min(i,j) < min(k,l) < max(i,j) < max(k,l):  
                    return False
        return True


    def read_conllu_file(self, path: str, inference=False) -> list:
        """
        Reads a file in CoNLLU format and converts it into a list of tree structures.

        The CoNLLU format is a standard for representing linguistic annotations, 
        especially for dependency parsing. Each sentence in the file is separated 
        by blank lines, and this method splits the file into individual sentences. 
        Each sentence is then converted into a tree structure using the 
        `conllustr2tree` method.

        Parameters:
            path (str): The file path of the CoNLLU file to be read.

            inference (bool): If True, the head and dependency relation (deprel) for each token 
                          are not assigned during parsing. This is useful when these 
                          attributes are to be inferred or predicted later. If False, 
                          these attributes are assigned based on the CoNLLU input.
            
        Returns:
            List: A list of tree structures, where each tree corresponds to a sentence 
                in the CoNLLU file. Each tree is a list of Token objects that represent 
                the words and annotations in the sentence.
        """
        trees = []
        with open(path) as f:
            sentences = f.read().strip().split("\n\n")
            for sent in sentences:
                tree = self.conllustr2tree(sent, inference)
                trees.append(tree)
        return trees
    
    def remove_non_projective_trees(self, trees: list['Token']) -> list:
        return [tree for tree in trees if self._is_projective(tree)]
    
    def write_conllu_file(self, path: str, trees: list['Token']) -> None:
        """
        Writes a list of tree structures to a file in CoNLLU format.

        CoNLLU format is a standard for representing linguistic annotations, 
        particularly for dependency parsing. This function converts each tree 
        structure (a list of Token objects representing a sentence) into CoNLLU 
        formatted strings and writes them to a file, separating sentences with blank lines.

        Parameters:
            trees (list): A list of tree structures, where each tree corresponds to a sentence.
                        Each tree is a list of Token objects that represent the words and 
                        annotations in the sentence.
            path (str): The file path where the CoNLLU file will be written.

        Returns:
            None: The function writes to a file and does not return any value.
        """
        with open(path, 'w') as f:
            for tree in trees:
                conllu_str = self.tree2conllustr(tree)
                f.write(conllu_str + "\n\n")

    


if __name__ == "__main__":

    reader = ConlluReader()

    conllustr = """
                1	Distribution	distribution	NOUN	S	Number=Sing	7	nsubj	_	_
                2	of	of	ADP	E	_	4	case	_	_
                3	this	this	DET	DD	Number=Sing|PronType=Dem	4	det	_	_
                4	license	license	NOUN	S	Number=Sing	1	nmod	_	_
                5	does	do	AUX	VM	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	7	aux	_	_
                6	not	not	PART	PART	Polarity=Neg	7	advmod	_	_
                7	create	create	VERB	V	Mood=Ind|Number=Plur|Tense=Pres|VerbForm=Fin	0	root	_	_
                8	an	a	DET	RI	Definite=Ind|Number=Sing|PronType=Art	12	det	_	_
                9	attorney	attorney	NOUN	S	Number=Sing	12	nmod	_	_
                10	-	-	PUNCT	FF	_	9	punct	_	_
                11	client	client	NOUN	S	Number=Sing	9	compound	_	_
                12	relationship	relationship	NOUN	S	Number=Sing	7	obj	_	_
                13	.	.	PUNCT	FS	_	7	punct	_	_
                    """

    print("Converting CoNLLU string to tree structure using conllustr2tree function and inference=False")
    # Convert the CoNLLU format string back to a tree structure
    converted_tree = reader.conllustr2tree(conllustr, inference=False)
    print("\nConverted Tree Structure:", converted_tree)
    print()

    print("Printing each token in the converted tree structure:")
    for token in converted_tree:
        print(token)
    print ()
    print("Converting CoNLLU string to tree structure using conllustr2tree function and inference=True")
        # Convert the CoNLLU format string back to a tree structure
    converted_tree = reader.conllustr2tree(conllustr, inference=True)
    print("\nConverted Tree Structure:", converted_tree)
    print()

    print("Printing each token in the converted tree structure:")
    for token in converted_tree:
        print(token)

    # Example tree structure (list of Token objects)
    tree = [
            Token(0, "ROOT", "ROOT", "_", "_", "_", "_", "_"),
            Token(1, "The", "the", "DET", "_", "Definite=Def|PronType=Art", 2, "det"),
            Token(2, "cat", "cat", "NOUN", "_", "Number=Sing", 4, "nsubj"),
            Token(3, "is", "be", "AUX", "_", "Mood=Ind|Tense=Pres|VerbForm=Fin", 4, "cop"),
            Token(4, "sleeping", "sleep", "VERB", "_", "VerbForm=Ger", 0, "root"),
            Token(5, ".", ".", "PUNCT", "_", "_", 4, "punct")
        ]


    # Accessing a specific element
    # For example, access the second token (which represents "The")
    second_token = tree[1]
    # Print the entire token
    print("Second Token: ", second_token)

    # Access and print specific attributes of the second token
    print("ID:", second_token.id)
    print("Form:", second_token.form)
    print("Lemma:", second_token.lemma)
    print("UPOS (Universal Part-of-Speech tag):", second_token.upos)
    print("CPOS (Language-specific part-of-speech tag):", second_token.cpos)
    print("Feats (List of morphological features):", second_token.feats)
    print("Head (Head of the current word):", second_token.head)
    print("Dep (Dependency relation to the head):", second_token.dep)
    print()
    # You can access other elements and their attributes in a similar way


    print("Converting tree structure to CoNLLU format string using tree2conllustr function")
    # Convert the tree to a CoNLLU format string
    conllustr = reader.tree2conllustr(tree)
    print("CoNLLU Format String:")
    print(conllustr)
    print()

    print("Reading a whole file in CoNLLU format using read_conllu_file function")
    # Read a whole file in CoNLLU format
    file_name = "datasets/en_partut-ud-dev_clean.conllu"
    sentence_trees = reader.read_conllu_file(path=file_name)
    print(f"Read a total of {len(sentence_trees)} sentences")

    print("Reading a whole file in CoNLLU format using read_conllu_file function")
    # Read a whole file in CoNLLU format
    file_name = "datasets/en_partut-ud-dev_clean.conllu"
    sentence_trees = reader.read_conllu_file(path=file_name)
    print(f"Read a total of {len(sentence_trees)} sentences")

    print ("Writting  the trees back to a file")
    file_name = "datasets/dummy_written.conllu"
    reader.write_conllu_file(file_name, sentence_trees)
