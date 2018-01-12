import math, random

train_size = 0.6
validation_size = 0.2
test_size = 0.2

train = []
test = []
validation = []
classes_num = 0
classes = []  # lista com as classes
attributes = []
attributes_type = {}  # 0 for nominal e 1 for numeric
nominal = 0
numeric = 1
attributes_values = {}  # dic de listas, cada lista i contém os valores do atributo i


class Node:
    def __init__(self):
        self.children = []
        self.values = []
        self.att = -1
        self.clas = -1

    def setAtt(self, a_index):
        self.att = a_index
        self.values = [val for val in attributes_values[a_index]]
        return

    def setClass(self, cls):
        self.clas = cls
        return


class Tree:
    def __init__(self):
        self.root = Node()


def read_benchmark():
    benchmark_in = open('benchmark.txt', 'rU').read().splitlines()

    global classes
    global classes_num
    classes_num = 2
    classes = ['Sim', 'Nao']

    for i in range(benchmark_in.__len__()):
        benchmark_in[i] = benchmark_in[i].split(';')

    titles = benchmark_in.pop(0)
    global attributes
    attributes = [titles[i] for i in range(0, 4)]

    for i,att in enumerate(attributes):
        vals = []
        for val in [inst[i] for inst in benchmark_in]:
            if val not in vals:
                vals.append(val)
        attributes_type[att] = nominal
        attributes_values[att] = vals

    for inst in benchmark_in:
        train.append(({att: inst[j] for j, att in enumerate(attributes)}, inst[4]))

    benchmark_in.clear()

    return


def info(dj):
    sum = 0.0
    for i in classes:
        probi = 0.0
        for element in dj:
            if element[1] == i:
                probi += 1.0
        if dj.__len__() != 0:
            probi = probi/dj.__len__()

        if probi != 0:
            sum += probi * math.log(probi, 2)

    return -1*sum


def infoa(d, a_index):
    sum = 0.0
    for j in attributes_values[a_index]:
        dj = []
        for element in d:
            if element[0][a_index] == j:
                dj.append(element)
        dj_norm = dj.__len__()/d.__len__()

        sum += dj_norm * info(dj)

    return sum


def gain(d, a_index):
    return info(d) - infoa(d, a_index)


def bestAtt(d, l):
    attributes_scores = []
    for a in l:
        attributes_scores.append(gain(d, a))
    return l[attributes_scores.index(max(attributes_scores))]


def induction(d, l):
    n = Node()   # 1

    single_class = True  # 2
    classes_freq = {cls: 0 for cls in classes}
    yi = d[0][1]
    for element in d:
        classes_freq[element[1]] += 1
        if element[1] != yi:
            single_class = False

    if single_class is True: # 2
        n.setClass(yi)
        return n

    if l.__len__() == 0:  # 3
        n.setClass(classes_freq.index(max(classes_freq)))
        return n

    a = bestAtt(d, l)  # 4.1
    n.setAtt(a)  # 4.2
    l.remove(a)  # 4.3

    for v in attributes_values[a]:
        dv = [instance for instance in d if instance[0][a] == v]  #  4.4.1

        if dv.__len__() == 0:
            leaf = Node()
            leaf.setClass(classes_freq.index(max(classes_freq)))
            n.children.append(leaf)
        else:
            n.children.append(induction(dv, l))

    return n


def classifier(node: Node, instance: list):   # instance has only attributes, not a tuple
    if node.att != -1:
        att = node.att
        if attributes_type[att] == nominal:
            for i, val in enumerate(node.values):
                if val == instance[att]:
                    return classifier(node.children[i], instance)

        elif attributes_type[att] == numeric:
            if instance[att] <= node.values:
                return classifier(node.children[0], instance)
            else:
                return classifier(node.children[1], instance)
    else:
        return node.clas

    print('valor de atributo não encontrado')
    return


if __name__ == '__main__':
    read_benchmark()

    tree = Tree()
    d = train
    l = attributes
    tree.root = induction(d, l)
    print(classifier(tree.root, d[0][0]))
    print("haha")
