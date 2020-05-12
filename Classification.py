import numpy as np

np.random.seed(42)

chi_table = {0.01: 6.635,
             0.005: 7.879,
             0.001: 10.828,
             0.0005: 12.116,
             0.0001: 15.140,
             0.00001: 19.511}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.
    """
    # count how many 1 labels we have
    trueLabels = np.count_nonzero(data[:, -1])
    # if all labels are the same, the impurity is 0
    if (trueLabels == 0 or trueLabels == data.shape[0]):
        return 0
    else:
        # calculate the impurity
        falseLables = len(data) - trueLabels
        gini = (trueLabels / len(data)) ** 2 + (falseLables / len(data)) ** 2
        return 1 - gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.
    """
    # count how many 1 labels we have
    trueLabels = np.count_nonzero(data[:, -1])
    # if all labels are the same, the impurity is 0
    if trueLabels == 0 or trueLabels == data.shape[0]:
        return 0
    else:
        # calculate the impurity, using log 2 as seen in class
        falseLables = len(data) - trueLabels
        entropy_true = (trueLabels / len(data)) * np.log2(trueLabels / len(data))
        entropy_false = (falseLables / len(data)) * np.log2(falseLables / len(data))
        return 0 - (entropy_true + entropy_false)


class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic
    # functionality as described in the notebook. It is highly recommended that you
    # first read and understand the entire exercise before diving into this class.

    def __init__(self, feature, value, data):
        self.feature = feature
        self.value = value
        self.data = data
        self.children = []
        self.classification = -1
        self.parent = None

    def add_child(self, node):
        self.children.append(node)


def build_tree(data, impurity, p_value=1):
    """
    Build a tree using the given impurity measure and training dataset.
    You are required to fully grow the tree until all leaves are pure.

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    # check if there is no data
    if data.shape[0] < 1:
        return
    # checks if the dataset isn't pure
    if np.unique(data[:, -1]).size > 1:
        bestFeature,  BestFeatureList = calcBestFeature(data, impurity)
        # create a node at set it's classification
        root = DecisionNode(bestFeature, BestFeatureList[2], data)
        if np.sum(data[:, -1]) > (data.shape[0] / 2):
            root.classification = 1.0
        else:
            root.classification = 0
        if p_value == 1 or chi2(data,  BestFeatureList) >= chi_table[p_value]:
            # add children to root - (leftChild <= value, rightChild > value)
            children = BestFeatureList[1]
            leftChild = build_tree(children[0], impurity, p_value)
            leftChild.parent = root
            root.add_child(leftChild)
            RightChild = build_tree(children[1], impurity, p_value)
            RightChild.parent = root
            root.add_child(RightChild)
            # set calssification of node according to majority
        elif np.count_nonzero(data[:, -1]) > (data.shape[0]/2):
            root.classification = 1.0
        else:
            root.classification = 0
    else:
        # the dataset is pure so we set this node as a leaf
        root = DecisionNode(feature='leaf', value=data.shape[0], data=data)
        root.classification = data[0][-1]
    return root


# this function calculates the best feature
def calcBestFeature(data, impurity):
    bestFeature = 0
    BestFeatureList = (0, None, None)
    for i in range(data.shape[1] - 1):
        featureList = calc_gain(data, i, impurity)
        if featureList[0] > BestFeatureList[0]:
            bestFeature = i
            BestFeatureList = featureList
    return bestFeature, BestFeatureList


def calc_gain(data, feature, impurity):
    values = np.unique(data[:, feature])
    thresholds = []
    for i in range(len(values) - 1):
        thresholds.append((values[i] + values[i + 1]) / 2)
    impurityCalc = impurity(data)
    minGain = np.infty
    bestSplit = {}
    bestThreshold = 0
    # we find the best gain, and split data according to the threshold
    for threshold in thresholds:
        split = {0: data[data[:, feature] <= threshold], 1: data[data[:, feature] > threshold]}
        leftSplitGain = split[0].shape[0] / data.shape[0] * impurity(split[0])
        rightSplitGain = split[1].shape[0] / data.shape[0] * impurity(split[1])
        totalGain = leftSplitGain + rightSplitGain
        if totalGain < minGain:
            minGain = totalGain
            bestSplit = split
            bestThreshold = threshold
    return impurityCalc - minGain, bestSplit, bestThreshold


def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    if node is None:
        return
    while node.children:
        # left child if smaller or equal to the threshold
        if instance[node.feature] <= node.value:
            node = node.children[0]
        # right child if larger than the threshold
        else:
            node = node.children[1]
    return node.classification


def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    labelList = np.array([])
    counter = 0
    for instance in dataset:
        prediction = predict(node, instance)
        if (prediction == instance[-1]):
            labelList = np.append(labelList, True)
            counter = counter + 1
        else:
            labelList = np.append(labelList, False)
    return (sum(labelList) / len(labelList)) * 100


# this is the chi square function
def chi2(data, BestFeatureList):
    p_true = (np.sum(data[:, -1])) / data.shape[0]
    p_false = 1 - p_true
    x_square = 0
    splits = BestFeatureList[1]
    for split in splits.values():
        df = split.shape[0]
        nf = np.sum(split[:, -1])
        pf = df - nf
        e0 = df * p_false
        e1 = df * p_true
        x_square += (np.square(pf - e0) / e0) + (np.square(nf - e1) / e1)
    # calculates as seen in class
    return x_square


# this is the post pruning function
def post_pruning(root, dataset, removed, internal):
    if root:
        removed.append(calc_accuracy(root, dataset))
        if not root.children:
            internal.append(0)
        else:
            leavesOfParent = []
            numOfInternal = [0]
            update_plist(root, leavesOfParent, numOfInternal)
            internal.append(numOfInternal[0])
            chosenParent = None
            accuracy = -1
            for parent in leavesOfParent:
                tmpChildren = parent.children
                parent.children = []
                if np.sum(parent.data[:, -1]) > (parent.data.shape[0] / 2):
                    parent.classification = 1.0
                else:
                    parent.classification = 0
                curAccuracy = calc_accuracy(root, dataset)
                parent.children = tmpChildren
                if curAccuracy > accuracy:
                    accuracy = curAccuracy
                    chosenParent = parent
            chosenParent.children = []
            post_pruning(root, dataset, removed, internal)


def update_plist(node, leavesOfParent, internal):
    if node:
        if node.children:
            internal[0] += 1
            update_plist(node.children[0], leavesOfParent, internal)
            update_plist(node.children[1], leavesOfParent, internal)
        elif node.parent not in leavesOfParent:
                leavesOfParent.append(node.parent)



def print_tree(node):
    '''
    prints the tree according to the example in the notebook

    Input:
    - node: a node in the decision tree

    This function has no return value
    '''
    spaces = 0
    toStr(node, spaces)


def toStr(node, spaces):
    if node.children:
        print((" " * spaces), "[X", node.feature, "<=", node.value, "]")
    else:
        print((" " * spaces), "leaf: [", {node.classification: node.value}, "]")

    for child in node.children:
        toStr(child, spaces + 4)