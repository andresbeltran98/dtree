import enum
import math


CLASS_NAME = 'CLASS'
type_key = 'type'
choices_key = 'choices'


class AttrType(enum.Enum):
    BIN = 'Binary'
    NOM = 'Nominal'
    CONT = 'Continuous'
    CL = CLASS_NAME


class Criteria(enum.Enum):
    IG = 'InformationGain'
    GR = 'GainRatio'
    GINI = 'Gini'


class Attribute:
    def __init__(self, type, name, value):
        self.name = name
        self.type = type
        self.value = value

    def __str__(self):
        if self.type == AttrType.CONT:
            return self.name + ' (split value = ' + str(self.value) + ')'
        return self.name

    def __repr__(self):
        return self.__str__()


def attr_type(attr_dict, attr_name):
    return attr_dict[attr_name][type_key]


def attr_choices(attr_dict, attr_name):
    return attr_dict[attr_name][choices_key]


def split_vals(df, cont_attr_name):
    """
    Returns the split values (test values) for a numeric attribute
    :param df: initial dataset
    :param cont_attr_name: name of the continuous attribute
    :return: list of split values
    """
    df_size = df.shape[0]

    if df_size == 0:
        return []

    if df_size == 1:
        return [df[cont_attr_name][0]]

    # Sort values of col "attr_name" in ascending order
    dfs = df.sort_values(cont_attr_name).reset_index(drop=True)
    prev_num = dfs[cont_attr_name][0]
    prev_class = dfs[CLASS_NAME][0]
    test_values = []

    fill_splt_vals(dfs, df_size, cont_attr_name, test_values, start_idx=0, prev_num=prev_num, prev_class=prev_class)
    return test_values


def fill_splt_vals(dfs, size, cont_attr_name, test_values, start_idx=0, prev_num=0, prev_class=0):
    """
    Split values for which the class labels are different on both sides
    :param dfs:
    :param size:
    :param cont_attr_name:
    :param test_values:
    :param start_idx:
    :param prev_num:
    :param prev_class:
    :return:
    """

    counter = [0, 0]
    i = start_idx
    cur_class = dfs[CLASS_NAME][i]
    cur_num = dfs[cont_attr_name][i]

    # Duplicates
    while (i+1) < size and dfs[cont_attr_name][i] == dfs[cont_attr_name][i+1]:
        counter[dfs[CLASS_NAME][i]] += 1
        i += 1

    counter[dfs[CLASS_NAME][i]] += 1
    next_start_idx = i+1

    if next_start_idx == size:
        if prev_num == cur_num:
            return

        if (counter[0] > 0 and counter[1] > 0) or (counter[not prev_class] != 0):
            test_values.append((prev_num + cur_num)/2)
    else:
        next_num = dfs[cont_attr_name][next_start_idx]

        if counter[0] > 0 and counter[1] > 0:
            if prev_num != cur_num:
                test_values.append((prev_num + cur_num)/2)

            test_values.append((cur_num + next_num)/2)
            fill_splt_vals(dfs, size, cont_attr_name, test_values, start_idx=next_start_idx, prev_num=next_num)
            return

        if (prev_num != cur_num) and (counter[not prev_class] != 0):
            test_values.append((prev_num+cur_num)/2)

        fill_splt_vals(dfs, size, cont_attr_name, test_values, start_idx=next_start_idx, prev_num=cur_num, prev_class=cur_class)


def choose_attribute(df, attr_dict, criteria):
    dict = {}

    if criteria == Criteria.IG or criteria == Criteria.GR:
        entropy_class = entropy_of(df, attr_dict, CLASS_NAME)

    for attr_name in attr_dict:
        type = attr_type(attr_dict, attr_name)

        if type == AttrType.BIN or type == AttrType.NOM:
            new_attr = Attribute(type, attr_name, None)
            if criteria == Criteria.IG:
                dict[new_attr] = information_gain(df, attr_dict, attr_name, entropy_class)
            elif criteria == Criteria.GR:
                dict[new_attr] = gain_ratio(df, attr_dict, attr_name, entropy_class)
            elif criteria == Criteria.GINI:
                dict[new_attr] = gini_impurity(df, attr_dict, attr_name)

        elif type == AttrType.CONT:
            split_pts = split_vals(df, attr_name)
            if not split_pts:
                continue

            optimal_splt_point = -1
            max_criteria = -1
            min_criteria = float('inf')

            for val in split_pts:
                criteria_val = 0
                if criteria == Criteria.IG:
                    criteria_val = information_gain(df, attr_dict, attr_name, entropy_class, is_cont=True, cont_val=val)
                elif criteria == Criteria.GR:
                    criteria_val = gain_ratio(df, attr_dict, attr_name, entropy_class, is_cont=True, cont_val=val)
                elif criteria == criteria.GINI:
                    criteria_val = gini_impurity(df, attr_dict, attr_name, is_cont=True, cont_val=val)
                    if criteria_val < min_criteria:
                        optimal_splt_point = val
                        min_criteria = criteria_val
                    continue

                if criteria_val > max_criteria:
                    optimal_splt_point = val
                    max_criteria = criteria_val

            new_attr = Attribute(type, attr_name, optimal_splt_point)
            dict[new_attr] = max_criteria

    # No attributes remaining
    if not dict:
        return None

    # Choose minimum value for Gini
    if criteria == Criteria.GINI:
        return min(dict, key=dict.get)

    # Choose max value for IG or GR
    return max(dict, key=dict.get)


def information_gain(df, attr_dict, attr_name, entropy_class, is_cont=False, cont_val=0):
    return entropy_class - entropy_class_given(df, attr_dict, attr_name, is_cont, cont_val)


def gain_ratio(df, attr_dict, attr_name, entropy_class, is_cont=False, cont_val=0):
    attr_entr = entropy_of(df, attr_dict, attr_name, is_cont, cont_val)
    if attr_entr == 0:
        return 0
    return information_gain(df, attr_dict, attr_name, entropy_class, is_cont, cont_val) / attr_entr


def gini_impurity(df, attr_dict, attr_name, is_cont=False, cont_val=0):
    if is_cont:
        return prob_cont(df, attr_name, cont_val, 0) * gini_index(df, attr_name, 0, is_cont=True, cont_val=cont_val) + \
               prob_cont(df, attr_name, cont_val, 1) * gini_index(df, attr_name, 1, is_cont=True, cont_val=cont_val)

    sum = 0
    choices = attr_choices(attr_dict, attr_name)
    for choice in choices:
        sum += prob(df, attr_name, choice) * gini_index(df, attr_name, choice)
    return sum


def gini_index(df, attr_name, choice, is_cont=False, cont_val=0):
    if is_cont:
        if choice:
            subset_df = df[df[attr_name] > cont_val]
        else:
            subset_df = df[df[attr_name] <= cont_val]
    else:
        subset_df = df[df[attr_name] == choice]

    p_true = prob(subset_df, CLASS_NAME, 1)
    p_false = prob(subset_df, CLASS_NAME, 0)
    return 1 - (p_false**2 + p_true**2)


def entropy_of(df, attr_dict, attr_name, is_cont=False, cont_val=0):
    """
    Entropy of a given attribute
    :param df:
    :param attr_dict:
    :param attr_name:
    :param is_cont:
    :param cont_val:
    :return:
    """
    sum = 0
    if is_cont:
        p = prob_cont(df, attr_name, cont_val, 0)
        if p != 0:
            sum += p * math.log2(p)
        q = prob_cont(df, attr_name, cont_val, 1)
        if q != 0:
            sum += q * math.log2(q)

        return -1 * sum

    for choice in attr_choices(attr_dict, attr_name):
        p = prob(df, attr_name, choice)
        if p != 0 and p != 1:
            sum += p * math.log2(p)

    return -1 * sum


def entropy_class_given(df, attr_dict, attr_name, is_cont=False, cont_val=0):
    """
    Returns the entropy of class=True given an attribute name
    """
    if is_cont:
        p = prob_cont(df, attr_name, cont_val, 0) * entropy(prob_y_given_cont(df, attr_name, cont_val, 0))
        q = prob_cont(df, attr_name, cont_val, 1) * entropy(prob_y_given_cont(df, attr_name, cont_val, 1))
        return p + q

    sum = 0
    choices = attr_choices(attr_dict, attr_name)
    for choice in choices:
        sum += prob(df, attr_name, choice) * entropy(prob_y_given(df, attr_name, choice))
    return sum


def entropy(x):
    if x == 0 or x == 1:
        return 0
    return -x*math.log2(x) - (1-x)*math.log2((1-x))


def prob(df, attr_name, choice):
    """
    Returns the probability that the attribute has a given value
    """
    try:
        selection = df[df[attr_name] == choice]
        return float(selection.shape[0] / df.shape[0])
    except:
        return 0


def prob_cont(df, attr_name, choice, cont_dir):
    """
    Same as prob() but for continuous attributes
    """
    try:
        if cont_dir:
            selection = df[df[attr_name] > choice]
        else:
            selection = df[df[attr_name] <= choice]

        return float(selection.shape[0] / df.shape[0])
    except:
        return 0


def prob_y_given(df, attr_name, choice):
    """
    Returns the probability that the attribute has a given value and has as True class label
    """
    try:
        selection = df[(df[attr_name] == choice) & (df[CLASS_NAME])]
        num_of_choice = df[(df[attr_name] == choice)]
        return float(selection.shape[0] / num_of_choice.shape[0])
    except:
        return 0


def prob_y_given_cont(df, attr_name, choice, cont_dir):
    """
    Same as prob_y_given() but for continuous attributes
    """
    try:
        if cont_dir:
            selection = df[(df[attr_name] > choice) & (df[CLASS_NAME])]
            num_of_choice = df[(df[attr_name] > choice)]
        else:
            selection = df[(df[attr_name] <= choice) & (df[CLASS_NAME])]
            num_of_choice = df[(df[attr_name] <= choice)]

        return float(selection.shape[0] / num_of_choice.shape[0])
    except:
        return 0


def is_pure_node(df):
    """
    Returns True if all class labels are the same class
    """
    return df.CLASS.nunique() == 1


def majority_class(df):
    """
    Returns the majority class label
    """
    return df.CLASS.value_counts().idxmax()


def partition_df(df, attr_name, value, is_cont=False, cont_dir=0):
    """
    Partitions data using attr_name and the given value.
    Returns a new df with the partitioned data
    """
    if is_cont and cont_dir:
        return df[df[attr_name] > value]
    if is_cont and not cont_dir:
        return df[df[attr_name] < value]
    return df[df[attr_name] == value]



