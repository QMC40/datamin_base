from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.tree import _tree, _utils
from sklearn.tree._classes import DecisionTreeClassifier

# def tree_to_code(tree, feature_names: List[]):
#     tree_ = tree.tree_
#     feature_name = [
#         feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
#         for i in tree_.feature
#     ]
#     print(f"def tree({', '.join(feature_names)}):")

#     def recurse(node, depth):
#         indent = "  " * depth
#         if tree_.feature[node] != _tree.TREE_UNDEFINED:
#             name = feature_name[node]
#             threshold = tree_.threshold[node]
#             print(f"{indent}if {name} <= {threshold}:")
#             recurse(tree_.children_left[node], depth + 1)
#             print(f"{indent}else:  # if {name} > {threshold}")
#             recurse(tree_.children_right[node], depth + 1)
#         else:
#             print(f"{indent}return {tree_.value[node]}")

#     recurse(0, 1)


def tree_to_decision_box_boundaries(
    tree: DecisionTreeClassifier,
    feat_data: List[Tuple[bool, str, int, int, Optional[int], Optional[np.ndarray]]],
) -> OrderedDict:

    tree_ = tree.tree_
    boundaries: List[Union[List[float], List[List[float]]]] = []
    for i in range(len(feat_data)):
        boundaries.append([])  # type: ignore[arg-type]

    def recurse(node: Any, depth: int) -> None:
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            id = tree_.feature[node]

            if not tree_.cat_pos[id]:
                threshold = tree_.threshold[node]
                boundaries[id].append(threshold)
            else:
                partition = tree_.partition[node]
                split_vals = _utils._get_cat_node_split(partition, tree_.cat_maxval[id])
                boundaries[id].append(list(split_vals))  # type: ignore[arg-type]

            recurse(tree_.children_left[node], depth + 1)
            recurse(tree_.children_right[node], depth + 1)

    recurse(0, 1)

    feature_names = [data[1] for data in feat_data]
    res_dict = OrderedDict({})

    # Clean-up
    for id in range(len(feat_data)):
        if feature_names[id] == "Sensitive":
            continue
        # Cont. features
        if not tree_.cat_pos[id]:
            boundaries[id] = sorted(list(set(boundaries[id])))  # type: ignore
        # Cat features - tuples to index
        else:
            tuples = list(zip(*boundaries[id]))  # type: ignore[arg-type]
            unique_sets = set(tuples)
            size = len(unique_sets)
            if size > 0:
                mapping = dict(zip(unique_sets, range(len(unique_sets))))
                # boundaries[id] = [mapping[t] for t in tuples]
                boundaries[id] = [mapping[t] for t in tuples]  # type: ignore[assignment]
            else:
                tot_size = feat_data[id][3] - feat_data[id][2]
                boundaries[id] = [0.0] * tot_size

        res_dict[feature_names[id]] = boundaries[id]
    return res_dict


def save_box_encoding_json(
    path: str, enc_dict: Dict[str, List[float]], is_cat: List[int]
) -> None:
    m_str = "{"
    for i, (k, v) in enumerate(enc_dict.items()):
        is_c = i in is_cat
        m_str += f'"{k}": ' + '{"size": '
        if is_c:
            m_str += str(max(v) + 1) + ", "
            m_str += '"mapping": '
        else:
            m_str += str(len(v) + 1) + ", "
            m_str += '"borders": '
        m_str += str(v)
        m_str += "}, "
    m_str = m_str[:-2]
    m_str += "}"
    with open(path, "w") as fp:
        fp.write(m_str)
