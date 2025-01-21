from conllu_reader import ConlluReader
import math

class PostProcessor():
    """
    A class for post-processing syntactic trees parsed from CoNLL-U formatted files.

    This class provides functionality to correct trees that have issues such as 
    multiple roots or tokens without an assigned head. It ensures that each tree 
    conforms to a standard structure with a single root and all tokens having 
    an assigned head.

    Methods:
        postprocess: Corrects the trees in a given CoNLL-U file and returns the corrected trees.
    """

    def postprocess(self,path:str):
        """
        Post-processes the syntactic trees from a CoNLL-U file to correct structural issues.

        This method reads a CoNLL-U file, identifies and corrects trees with multiple roots 
        or tokens without an assigned head. It ensures each tree has a single root, and 
        all tokens have a designated head. Tokens without a head are assigned to the 
        root of their respective trees.

        Args:
            path (str): The file path of the CoNLL-U file to be processed.

        Returns:
            list: A list of corrected syntactic trees.
        """
        reader = ConlluReader()
        trees = reader.read_conllu_file(path, inference=False)

        for tree in trees:
            nodes_without_head = [idx for idx in range(1,len(tree))]
            nodes_to_root = set([])
            min_node_to_root = math.inf

            for token in tree[1:]:
                head, dep, dependent = token.head, token.dep, token.id
                if head != "_":
                    nodes_without_head.remove(dependent)

                if head == 0:
                    nodes_to_root.add(dependent)
                    if min_node_to_root > dependent:
                        min_node_to_root = dependent

            #If multi-root, we make it single root
            if len(nodes_to_root) > 1:
                for tokenid in nodes_to_root:
                    if tokenid != min_node_to_root:
                        tree[tokenid].head = min_node_to_root

            #If no word was assigned as the root, we make root the first token of the sentence that does not have an assigned head
            if len(nodes_to_root) == 0 and len(nodes_without_head) > 0:
                tree[nodes_without_head[0]].head = 0
                min_node_to_root = nodes_without_head[0]
                nodes_without_head.remove(nodes_without_head[0])

            #The rest of tokens that have not been assigned a head, are assigned as children of the root of the sentence
            for token in nodes_without_head:
                tree[token].head = min_node_to_root

                

        return trees
    

if __name__ == "__main__":

    p = PostProcessor()

    reader = ConlluReader()
    print ("An example of a corrupted tree, before postprocessing:")
    #This sentence has two node with the ROOT (0) as a head: Words 5 and 7
    #This sentence has also words that have not an assigned head: 6 and 8. 
    trees = reader.read_conllu_file("datasets/corrupted_output.conllu")
    for tree in trees:
        for token in tree:
            print (token)

    print()
    print()
    trees = p.postprocess("datasets/corrupted_output.conllu")
    print ("An example of a (now no) corrupted tree, after postprocessing:")
    for tree in trees:
        for token in tree:
            print (token)

