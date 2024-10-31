from pptree import Node, print_tree


class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []

        if parent:
            self.parent.children.append(self)

    def flatten(self):
        retval = [self.name]
        for child in self.children:
            retval.extend(child.flatten())
        return retval

    def find(self, x):
        if self.name == x:
            return self
        for node in self.children:
            n = node.find(x)
            if n:
                return n
        return None

    @property
    def depth(self):
        if self.parent:
            return self.parent.depth + 1
        return 0

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Node(name = {self.name}, parent = {self.parent.name if self.parent else 'None'})"


def flatten(A):
    rt = []
    for i in A:
        if isinstance(i, list):
            rt.extend(flatten(i))
        else:
            rt.append(i)
    return rt


class LabelTree:
    def __init__(self):
        self._root = Node("sample")

        cell = Node("cell", self._root)
        capillary = Node("capillary", self._root)
        buffer = Node("buffer", self._root)
        void = Node("void", self._root)

        membrane = Node("membrane", cell)
        nucleus = Node("nucleus", cell)
        mitochondria = Node("mitochondria", cell)
        chloroplast = Node("chloroplast", cell)
        lipid = Node("lipid", cell)
        vacuole = Node("vacuole", cell)

        er = Node("endoplasmic reticulum", cell)
        granule = Node("granule", cell)
        golgi = Node("golgi", cell)

        golgi = Node("symbiont", cell)

        nucleolus = Node("nucleolus", nucleus)
        heterochromatin = Node("heterochromatin", nucleus)
        euchromatin = Node("euchromatin", nucleus)


    def exisit(self, key):
        return self._root.find(key) is not None

    def pptree(self):
        print_tree(self._root)

    def flat_children(self, label=None):
        if label:
            return self._root.find(label).flatten()
        return self._root.flatten()

    def extract_materials(self, labels):

        nodes = [self._root.find(label) for label in labels]
        for n, l in zip(nodes, labels):
            if not n:
                self.pptree()
                raise ValueError(f"Couldn't find matching organelle {l}")

        # Overwrite ownership of organelle, starting from smallest depth
        organelle2label = dict()
        for i, node in enumerate(
            sorted(
                nodes,
                key=lambda x: (x.depth, x.name),
            )
        ):

            for key in node.flatten():
                organelle2label[key] = node.name

        # aggregate resulkts
        return [
            [k for k, v in organelle2label.items() if v == label] for label in labels
        ]

    def organelles_to_features(self, organelles):
        keys = flatten(organelles)
        values = self.extract_materials(keys)
        extracted = {k: v for k, v in zip(keys, values)}
        featurelist = []
        for label in organelles:
            if not isinstance(label, (list, tuple)):
                label = [label]
            labels = []
            for synonym in label:
                ext = extracted[synonym]
                labels += ext

            featurelist.append(labels)
        return featurelist


Organelles = LabelTree()


if __name__ == "__main__":
    # node = Organelles._root.find("nucleus")
    # print(node.name)
    # print(node.flatten())

    # print_tree(Organelles._root)
    # print_tree(node)
    features = Organelles.extract_materials(["cell", "chloroplast", "nucleus"])
