class Token():
    """
    A class to encode a line from a CoNLLU file.

    Attributes:
    - id (int): Token ID in the sentence.
    - form (str): The form or orthography of the word.
    - lemma (str): The lemma or base form of the word.
    - upos (str): Universal part-of-speech tag.
    - cpos (str): Language-specific part-of-speech tag; not used in this assignment.
    - feats (str): List of morphological features from the universal feature inventory or language-specific extension; separated by '|'.
    - head (int): Head of the current token, which is either a value of ID or zero ('0').
    - dep (str): Universal dependency relation to the HEAD.
    - deps (str): Enhanced dependency graph in the form of a list of head-deprel pairs.
    - misc (str): Any other annotation.

    Methods:
    - get_fields_list(): Returns a list of all the attribute values of the token, in the same order requited by the conllu format

    # Example usage:

    token_example = Token(1, "Distribution", "distribution", "NOUN", "S", "Number=Sing", 7, "nsubj")
    """

    def __init__(self, id, form, lemma, upos, cpos, feats, head, dep):

        self._id = id
        self._form = form
        self._lemma = lemma
        self._upos = upos
        self._cpos = cpos
        self._feats = feats
        self._head = head
        self._dep = dep
        self._deps = "_"  # Not used in our assignment
        self._misc = "_"  # Not used in our assignment

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        # Add validation or transformation if necessary
        self._id = value

    @property
    def form(self):
        return self._form

    @form.setter
    def form(self, value):
        # Add validation or transformation if necessary
        self._form = value

    @property
    def lemma(self):
        return self._lemma

    @lemma.setter
    def lemma(self, value):
        # Add validation or transformation if necessary
        self._lemma = value

    @property
    def upos(self):
        return self._upos

    @upos.setter
    def upos(self, value):
        # Add validation or transformation if necessary
        self._upos = value

    @property
    def cpos(self):
        return self._cpos

    @cpos.setter
    def cpos(self, value):
        # Add validation or transformation if necessary
        self._cpos = value

    @property
    def feats(self):
        return self._feats

    @feats.setter
    def feats(self, value):
        # Add validation or transformation if necessary
        self._feats = value

    @property
    def head(self):
        return self._head

    @head.setter
    def head(self, value):
        # Add validation or transformation if necessary
        self._head = value

    @property
    def dep(self):
        return self._dep

    @dep.setter
    def dep(self, value):
        # Add validation or transformation if necessary
        self._dep = value

    @property
    def deps(self):
        return self._deps

    @deps.setter
    def deps(self, value):
        # Add validation or transformation if necessary
        self._deps = value

    @property
    def misc(self):
        return self._misc

    @misc.setter
    def misc(self, value):
        # Add validation or transformation if necessary
        self._misc = value

    def get_fields_list(self):
        return [self.id, self.form, self.lemma, self.upos, self.cpos, self.feats, self.head, self.dep, self.deps, self.misc]
    

    def __str__(self):
            """
            Returns a string representation of the Token object with all attributes separated by tabs.

            Returns:
                str: A tab-separated string of all the attributes.
            """
            return "\t".join(map(str, [self.id, self.form, self.lemma, self.upos, self.cpos, self.feats, self.head, self.dep, self.deps, self.misc]))
    

