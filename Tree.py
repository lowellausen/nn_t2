import math, random, statistics, sys

train_size = 0.8  # proporções de treino/validação e teste
test_size = 0.2
k = 10  # k folds para o cross-val

ntree_vals = [10, 25, 50, 60]  # valores que serão usados nos cross-vals
m = 3  # valor m de amostragem de atributos. é setado(na leitura do dataset) como raiz da quant de atributos do prob

benchmark = False  # flag para imprimir as informaçãos quando for chamado o benchmark

dataset = []  # onde é lido o dataset inicialmente como uma lista de instancias (tuplas) do tipo (atributos, classe)
#                # atributos de uma instancia são sempre um dicionario onde a key é o nome do att
#                                                                                   e o valor, seu valor na instancia
train = []  # train e test seguem modelo do dataset
test = []
classes_num = 0  # onde será setado  o número de classes preditivas do dataset
classes = []  # lista com as classes
attributes = []  # lista com o nome de cada atributo do problema
attributes_type = {}  # dicionário onde a key é o nome do atributo e o valor o seu tipo. nominal ou numerical
nominal = 0
numerical = 1
attributes_values = {}  # dicT de listas, cada key é o nome do att e contem os valores do att

"""
Uma árvore é representada somente pelo seu nodo raíz. nodo é a única classe usada no trabalho
um nodo pode ser de atributo, que é um nodo interno que tem um filho para cada valor de atributo
ou um nodo de classe, que é um nodo folha que tem uma predição de classe
"""


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
        dataset.append(({att: inst[j] for j, att in enumerate(attributes)}, inst[4]))

    benchmark_in.clear()

    return


def read_haberman():
    haberman_in = open('haberman.data', 'rU').read().splitlines()

    global classes
    global classes_num
    classes_num = 2
    classes = ['1', '2']

    for i in range(haberman_in.__len__()):
        haberman_in[i] = haberman_in[i].split(',')

    titles = haberman_in.pop(0)
    global attributes
    attributes = [titles[i] for i in range(0, 3)]

    for i, att in enumerate(attributes):
        vals = []
        for val in [inst[i] for inst in haberman_in]:
            if val not in vals:
                vals.append(val)
        attributes_type[att] = numerical
        attributes_values[att] = vals

    for inst in haberman_in:
        dataset.append(({att: inst[j] for j, att in enumerate(attributes)}, inst[3]))

    haberman_in.clear()

    return


def read_cmc():
    cmc_in = open('cmc.data', 'rU').read().splitlines()

    global classes
    global classes_num
    classes_num = 3
    classes = ['1', '2', '3']

    for i in range(cmc_in.__len__()):
        cmc_in[i] = cmc_in[i].split(',')

    titles = cmc_in.pop(0)
    global attributes
    attributes = [titles[i] for i in range(0, 9)]

    for i, att in enumerate(attributes):
        vals = []
        for val in [inst[i] for inst in cmc_in]:
            if val not in vals:
                vals.append(val)
        if i == 0 or i == 3:
            attributes_type[att] = numerical
        else:
            attributes_type[att] = nominal
        attributes_values[att] = vals

    for inst in cmc_in:
        dataset.append(({att: inst[j] for j, att in enumerate(attributes)}, inst[9]))

    cmc_in.clear()

    return


def read_wine():
    wine_in = open('wine.data', 'rU').read().splitlines()

    global classes
    global classes_num
    classes_num = 3
    classes = ['1', '2', '3']

    for i in range(wine_in.__len__()):
        wine_in[i] = wine_in[i].split(',')

    titles = wine_in.pop(0)
    global attributes
    attributes = [titles[i] for i in range(1, 14)]

    for i, att in enumerate(attributes):
        vals = []
        for val in [inst[i+1] for inst in wine_in]:
            if val not in vals:
                vals.append(val)

        attributes_type[att] = numerical
        attributes_values[att] = vals

    for inst in wine_in:
        dataset.append(({att: inst[j+1] for j, att in enumerate(attributes)}, inst[0]))

    wine_in.clear()

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
    if attributes_type[a_index] == nominal:
        for j in attributes_values[a_index]:
            dj = []
            for element in d:
                if element[0][a_index] == j:
                    dj.append(element)
            dj_norm = dj.__len__()/d.__len__()

            sum += dj_norm * info(dj)

    elif attributes_type[a_index] == numerical:
        d1 = []
        d2 = []
        for element in d:
            if float(element[0][a_index]) <= attributes_values[a_index][0]:
                d1.append(element)
            else:
                d2.append(element)
        d1_norm = d1.__len__() / d.__len__()
        d2_norm = d2.__len__() / d.__len__()

        sum = (d1_norm * info(d1)) + (d2_norm * info(d2))

    return sum


def gain(d, a_index):
    return info(d) - infoa(d, a_index)


def attSampling(l):
    if l.__len__() > m:
        return random.sample(l, m)
    else:
        return l


def bestAtt(d, l):
    attributes_scores = []
    for a in l:
        attributes_scores.append(gain(d, a))

    if benchmark:
        print("Ganho de informação: " + str(max(attributes_scores)))
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
        n.setClass(max(classes_freq, key=lambda key: classes_freq[key]))
        return n

    a = bestAtt(d, attSampling(l))  # 4.1   em vez de passar l passa uma amostragem aleatória de m elementos de l
    n.setAtt(a)  # 4.2
    l.remove(a)  # 4.3

    if attributes_type[a] == nominal:
        for v in attributes_values[a]:
            dv = [instance for instance in d if instance[0][a] == v]  #  4.4.1

            if dv.__len__() == 0:
                leaf = Node()
                leaf.setClass(max(classes_freq, key=lambda key: classes_freq[key]))
                n.children.append(leaf)
            else:
                n.children.append(induction(dv, l[:]))
    elif attributes_type[a] == numerical:
        d1 = []
        d2 = []
        for element in d:
            if float(element[0][a]) <= attributes_values[a][0]:
                d1.append(element)
            else:
                d2.append(element)

        if d1.__len__() == 0:
            leaf = Node()
            leaf.setClass(max(classes_freq, key=lambda key: classes_freq[key]))
            n.children.append(leaf)
        else:
            n.children.append(induction(d1, l[:]))

        if d2.__len__() == 0:
            leaf = Node()
            leaf.setClass(max(classes_freq, key=lambda key: classes_freq[key]))
            n.children.append(leaf)
        else:
            n.children.append(induction(d2, l[:]))

    return n


def classifier(node: Node, instance: list):   # instance has only attributes, not a tuple
    if node.att != -1:
        att = node.att
        if attributes_type[att] == nominal:
            for i, val in enumerate(node.values):
                if val == instance[att]:
                    return classifier(node.children[i], instance)

        elif attributes_type[att] == numerical:
            if float(instance[att]) <= node.values[0]:
                return classifier(node.children[0], instance)
            else:
                return classifier(node.children[1], instance)
    else:
        return node.clas

    print('valor de atributo não encontrado')
    return


def genBootstraps(ntree, l):
    bootstraps = []
    for i in range(ntree):
        bootstrap = ([], [])  #  uma tupla com treino e teste
        bootstrap[0].extend(random.choices(l, k=l.__len__()))
        bootstrap[1].extend([inst for inst in l if inst not in bootstrap[0]])
        bootstraps.append(bootstrap)

    return bootstraps


def majorityVoting(forest, instance):
    votes = {cls: 0 for cls in classes}
    for tree in forest:
        votes[classifier(tree, instance)] += 1

    return max(votes, key=lambda key: votes[key])


def testForest(forest, test_data):   # testa a floresta no conjunto dado e retorna a acurácia
    acc = 0.0
    for inst in test_data:
        if inst[1] == majorityVoting(forest, inst[0]):
            acc += 1

    return acc/test_data.__len__()


def splitDataset():
    size = dataset.__len__()

    for i in range(int(train_size*size)):
        index = random.randint(0, dataset.__len__()-1)
        train.append(dataset.pop(index))

    test.extend(dataset)

    dataset.clear()

    return


def resolve(att):
    data = train[:]
    data = sorted(data, key=lambda x: x[0][att])

    results = {}  # dict com valor de corte: entropia

    for i in range(data.__len__()-1):
        if data[i][1] != data[i+1][1]:  # instancia consecutivas com calss divergente
            mean = (float(data[i][0][att]) + float(data[i+1][0][att]))/2

            if mean not in list(results.keys()):
                attributes_values[att] = [mean]  # temporariamente seta esse valor para facilitar o uso da func infoa
                results[mean] = infoa(data, att)  # avalia o corte encontrado em t0do o conj de treino

    attributes_values[att] = [min(results, key=lambda key: results[key])]

    return


def resolve_numericals():
    for att in attributes:
        if attributes_type[att] == numerical:
            resolve(att)

    return


def genFolds(l):    # divide o conjunto de treino em k folds
    #  fold_size = int(math.ceil(train.__len__()/k))
    l_size = l.__len__()
    folds = []

    for i in range(k):
        fold = []
        for j in range(l_size//k):
            if l.__len__() != 0:
                fold.append(l.pop(0))
        folds.append(fold)

    if l.__len__() != 0 :   # sobrou algum item porque a lista não é divisivel igualmente por k
        folds[k-1].extend(l)  # botamos o resto de l no último fold

    return folds
    #  return [train[i:i + fold_size] for i in range(0, train.__len__(), fold_size)]


def single_cross(ntree, folds):   # realiza o cross validaton para um conjunto de parâmetros. retorna médio e dp
    accuracies = []
    for i in range(k):
        cross_test = folds[i]

        cross_train = []
        for f in range(k):  # gera conjunto de treino com os folds que não são o de teste
            if f != i:
                cross_train.extend(folds[f])

        bootstraps = genBootstraps(ntree, cross_train)
        forest = [induction(bootstrap[0], attributes[:]) for bootstrap in bootstraps]

        accuracies.append(testForest(forest, cross_test))

    return statistics.mean(accuracies), statistics.pstdev(accuracies)


def cross_validation():
    folds = genFolds(train[:])
    results = {}  # dict ntree: (media, desvio)       media e desvio de accuracy

    for ntree in ntree_vals:
        results[ntree] = single_cross(ntree, folds)

    ntree = max(results, key=lambda key: results[key][0])

    out_name = sys.argv[1]
    for ntree in ntree_vals:
        out_name = out_name + str(ntree) + '_'
    out = open(out_name + '.csv', 'w')

    out.write("ntree;media;desvio\n")
    for ntree in ntree_vals:
        out.write(str(ntree) + ';' + str(results[ntree][0]) + ';' + str(results[ntree][1]) + '\n')

    out.close()

    bootstraps = genBootstraps(ntree, train)    # treina uma floresta usando o valor ótimo e com conj treino inteiro
    forest = [induction(bootstrap[0], attributes[:]) for bootstrap in bootstraps]

    return forest


def print_tree(node: Node, gap):   # instance has only attributes, not a tuple
    if node.att != -1:
        print(gap + "Nodo de atributo: " + str(node.att))
        for i in range(node.values.__len__()):
            print_tree(node.children[i], gap + "Nodo de atributo: " + str(node.att) + " " + str(node.att) + str(node.values[i]) + "---->")
    else:
        print(gap + "Nodo de classe: " + str(node.clas))

    return


if __name__ == '__main__':
    #  o programa deve obrigatoriamente receber um parametro que e o arquivos de dados a ser utilizado
    # opções:
    # - benchmark: gera uma unica arvore de benchmark, imprimindo o ganho em cada node split, e a árvore final
    # - haberman/cmc/wine: gera um floresta aleatória usando 10 fold cross validation
    # os valores de ntree pode ser dados como parametros apos o nome do dataset ex: "py -3 Tree.py cmc 15 20 25 50 75"
    # irá fazer 5 cross-validations usando 15 20 25 50 e 75 como valores de ntree, gerando uma floresta final com o
    # valor ntree ótimo, também gerando um arquivo haberman15_20_25_50_75_.csv com média e desvio de acurácia
    # para cada valor de ntree. valores default são 10 25 50 60
    # ao final é impresso a acurácia medida com o modelo ótimo no conjunto de testes

    if sys.argv.__len__() > 1:
        if sys.argv[1] == "benchmark":
            read_benchmark()
            m = 4
            benchmark = True
            tree = induction(dataset[:], attributes[:])
            print_tree(tree, "")
        else:
            if sys.argv.__len__() > 2:
                ntree_vals = [int(sys.argv[i]) for i in range(2, sys.argv.__len__())]
            if sys.argv[1] == "haberman":
                read_haberman()
            if sys.argv[1] == "cmc":
                read_cmc()
            if sys.argv[1] == "wine":
                read_wine()

            random.seed("cidy")  # seed fixa em cidy, for no reason
            m = int(math.sqrt(attributes.__len__()))  # dica da definição

            splitDataset()  # divide treino e teste

            resolve_numericals()  # acha valor de corte para atributos continuos,de acordo com entropia no conj treino

            forest = cross_validation()  # gera uma floresta ótima usando crossval(aqui gera arquivom com medias e desv

            print(testForest(forest, test))  # imprime a acurácia dessa floresta aplicada ao conj de teste
