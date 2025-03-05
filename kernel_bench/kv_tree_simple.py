import numpy as np


class KVTreeNode:
    def __init__(self):
        self.parent = -1
        self.id = -1
        self.seqlen = 0
        self.num_children = 0
        self.requests = []


def split_token(tree_token_size, node_num):
    min_length = np.ones(node_num, dtype=int)
    remaining_length = tree_token_size - node_num
    split_points = np.random.randint(0, remaining_length + 1, node_num - 1)
    split_points = np.sort(split_points)
    remaining_segments = np.diff(
        np.concatenate(([0], split_points, [remaining_length]))
    )
    segments = remaining_segments + min_length
    return segments


def generate_new_tree(tree_token_size, node_num):
    segments = split_token(tree_token_size, node_num)

    tree_info = []

    for i in range(node_num):
        new_node = KVTreeNode()
        if i != 0:
            new_node.parent = np.random.randint(0, i)
            tree_info[new_node.parent].indegree += 1

        new_node.id = i
        new_node.seqlen = segments[i]
        tree_info.append(new_node)

    return tree_info


def write_tree_structure(tree_info: KVTreeNode, filepath):
    with open(filepath, "w") as file:
        file.write(f"{len(tree_info)}\n")
        for node in tree_info:
            file.write(f"{node.fa} {node.id} {node.seqlen} {node.indegree}\n")


def retrive_from_file(filepath):
    tree_info = []
    with open(filepath, "r") as file:
        num_nodes = int(file.readline().strip())
        for _ in range(num_nodes):
            line = file.readline().strip().split()
            fa, id, seqlen, indegree = map(int, line)
            node = KVTreeNode()
            node.parent = fa
            node.id = id
            node.seqlen = seqlen
            node.num_children = indegree

            tree_info.append(node)
    return tree_info
