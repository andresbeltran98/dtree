from utils import *
from copy import deepcopy


class TreeNode:
    """
    This class represents a node in a Decision Tree
    """
    def __init__(self, name, schema, partitions=None, is_leaf=False):
        self.name = name
        self.schema = schema
        self.partitions = partitions
        self.is_leaf = is_leaf

    def get_type(self):
        return self.schema[type_key]

    def get_choices(self):
        return self.schema[choices_key]

    def partition_of(self, val):
        """
        Returns the corresponding partition for a given value
        :param val: The value to look for
        :return: partition corresponding to val
        """
        node_type = self.get_type()

        if node_type == AttrType.BIN or node_type == AttrType.NOM:
            for part in self.partitions:
                if part.value == val:
                    return part

        elif node_type == AttrType.CONT:
            for part in self.partitions:
                if part.contains_cont_val(val):
                    return part

    def __str__(self):
        if self.is_leaf:
            return '(Leaf ' + str(self.name) + ')'
        else:
            return self.name + str(list(self.partitions))

    def __repr__(self):
        return self.__str__()


class Partition:
    """
    This class represents a partition for a given branch of a given node
    """
    def __init__(self, partition_type, value, data, dir=0, child_node=None):
        self.type = partition_type
        self.value = value  # boolean for Binary, String for Nominal, double for Continuous
        self.dir = dir      # For continuous, 0 means <= value, 1 means > value
        self.data = data
        self.child_node = child_node

    def contains_cont_val(self, cont_val):
        return (cont_val <= self.value and self.dir == 0) or (cont_val > self.value and self.dir == 1)

    def __str__(self):
        if self.type == AttrType.CONT and self.dir == 0:
            return '≤' + str(self.value) + '->' + str(self.child_node)
        elif self.type == AttrType.CONT and self.dir == 1:
            return '>' + str(self.value) + '->' + str(self.child_node)

        return str(self.value) + '->' + str(self.child_node)

    def __repr__(self):
        return self.__str__()


class DecisionTree:
    """
    This class represents a Decision Tree
    """
    def __init__(self):
        self.root = None

    def __str__(self):
        return str(self.root)

    def __repr__(self):
        return self.__str__()

    def predict(self, example_iloc):
        """
        Returns the prediction for a novel example
        :param example_iloc: new example (row of a dataframe)
        :return: prediction for the novel example
        """
        return self.__search_class_label(example_iloc, self.root)

    def __search_class_label(self, example, node):
        """
        Traverses the tree recursively
        :param example: example
        :param node: root
        :return: prediction of the reached leaf-node
        """
        if node.is_leaf:
            return node.name

        part = node.partition_of(example[node.name])
        return self.__search_class_label(example, part.child_node)

    def train(self, df, attr_dict, criteria):
        """
        Trains a new decision tree
        :param df: initial dataset
        :param attr_dict: information about attributes
        :param criteria: IG, GR, or GINI
        """
        if df.empty:
            return None
        self.root = self.__create_tree(df, attr_dict, criteria)

    def __create_tree(self, df, attr_dict, criteria):
        """
        Creates a new decision tree from an initial dataset
        :param df: initial dataset
        :param attr_dict: attribute information
        :param criteria: Split criteria (IG, GR, or Gini)
        :return: the root node of the tree
        """

        # Base Cases
        if is_pure_node(df):
            return TreeNode(df.iloc[0][CLASS_NAME], attr_dict[CLASS_NAME], is_leaf=True)

        # Choose best test attribute
        test_attr = choose_attribute(df, attr_dict, criteria)
        if test_attr is None:
            return TreeNode(majority_class(df), attr_dict[CLASS_NAME], is_leaf=True)

        attr_name = test_attr.name
        attr_continuous_val = test_attr.value

        # Create partitions
        partitions = set()
        type = attr_type(attr_dict, attr_name)
        new_dict = deepcopy(attr_dict)
        del new_dict[attr_name]

        # Create partitions depending on the attribute type
        if type == AttrType.BIN:
            choices = [True, False]

            for choice in choices:
                samples = partition_df(df, attr_name, choice)
                part = Partition(type, choice, samples)
                if samples.empty:
                    part.child_node = TreeNode(majority_class(df), attr_dict[CLASS_NAME], is_leaf=True)
                else:
                    part.child_node = self.__create_tree(samples, new_dict, criteria)

                partitions.add(part)

        elif type == AttrType.NOM:
            for choice in attr_choices(attr_dict, attr_name):
                samples = partition_df(df, attr_name, choice)
                part = Partition(type, choice, samples)

                if samples.empty:
                    part.child_node = TreeNode(majority_class(df), attr_dict[CLASS_NAME], is_leaf=True)
                else:
                    part.child_node = self.__create_tree(samples, new_dict, criteria)

                partitions.add(part)

        elif type == AttrType.CONT:
            choices = [0, 1]

            for choice in choices:
                samples = partition_df(df, attr_name, attr_continuous_val, is_cont=True, cont_dir=choice)
                part = Partition(type, attr_continuous_val, samples, dir=choice)

                if samples.empty:
                    part.child_node = TreeNode(majority_class(df), attr_dict[CLASS_NAME], is_leaf=True)
                else:
                    part.child_node = self.__create_tree(samples, new_dict, criteria)

                partitions.add(part)

        # Create a new TreeNode with the formed partitions
        return TreeNode(attr_name, attr_dict[attr_name], partitions=partitions)

