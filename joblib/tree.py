import collections.abc
from collections import namedtuple

# --- Path Utilities ---

class SequenceKey(namedtuple('SequenceKey', ['idx'])):
    """A key for sequences (lists, tuples) containing the index."""
    def __repr__(self): return f'[{self.idx}]'

class DictKey(namedtuple('DictKey', ['key'])):
    """A key for dicts containing the dictionary key."""
    def __repr__(self): return f'[{repr(self.key)}]'

def keystr(path):
    """Turns a path tuple into a single, readable string."""
    return "".join(map(str, path))

# Sentinel object to represent a leaf in the structure definition (treedef).
_LEAF = object()

class PyTreeDef:
    """A class to hold the structure of a pytree and provide a JAX-like string representation."""
    def __init__(self, internal_def):
        self._internal_def = internal_def

    def __eq__(self, other):
        if not isinstance(other, PyTreeDef): return NotImplemented
        return self._internal_def == other._internal_def

    def __repr__(self):
        """Returns a JAX-like string representation of the pytree structure."""
        return f"PyTreeDef({self._build_repr_str(self._internal_def)})"

    def _build_repr_str(self, treedef):
        """Recursively builds the string for the __repr__ method."""
        if treedef is _LEAF: return '*'
        node_type = treedef[0]
        if node_type in (list, tuple):
            children_str = ", ".join(self._build_repr_str(s) for s in treedef[1])
            if node_type is list: return f"[{children_str}]"
            return f"({children_str},)" if len(treedef[1]) == 1 else f"({children_str})"
        elif node_type is dict:
            children_str = [f"{repr(k)}: {self._build_repr_str(s)}" for k, s in zip(treedef[1], treedef[2])]
            return f"{{{', '.join(children_str)}}}"
        return "<unknown>"

def _flatten_recursive(tree, is_leaf, with_path, path):
    """A recursive helper to flatten a tree, with or without paths."""
    if is_leaf and is_leaf(tree):
        if with_path:
            return [(path, tree)], _LEAF
        else:
            return [tree], _LEAF

    leaves, child_defs = [], []
    if isinstance(tree, (list, tuple)):
        for i, sub_tree in enumerate(tree):
            sub_leaves, sub_def = _flatten_recursive(sub_tree, is_leaf, with_path, (*path, SequenceKey(i)))
            leaves.extend(sub_leaves)
            child_defs.append(sub_def)
        return leaves, (type(tree), child_defs)
    elif isinstance(tree, dict):
        for key in sorted(tree.keys()):
            sub_leaves, sub_def = _flatten_recursive(tree[key], is_leaf, with_path, (*path, DictKey(key)))
            leaves.extend(sub_leaves)
            child_defs.append(sub_def)
        return leaves, (dict, sorted(tree.keys()), child_defs)
    else:
        if with_path:
            return [(path, tree)], _LEAF
        else:
            return [tree], _LEAF

def _unflatten_recursive(treedef, leaves_iter):
    """A recursive helper to build a tree from a treedef and an iterator of leaves."""
    if treedef is _LEAF: return next(leaves_iter)
    node_type = treedef[0]
    if node_type in (list, tuple):
        return node_type(_unflatten_recursive(s, leaves_iter) for s in treedef[1])
    elif node_type is dict:
        children = [_unflatten_recursive(s, leaves_iter) for s in treedef[2]]
        return dict(zip(treedef[1], children))
    raise TypeError(f"Unsupported treedef type: {node_type}")

def tree_flatten(tree, is_leaf=None):
    """Flattens a pytree into a list of leaves and a PyTreeDef."""
    leaves, internal_def = _flatten_recursive(tree, is_leaf, False, ())
    return leaves, PyTreeDef(internal_def)

def tree_flatten_with_path(tree, is_leaf=None):
    """Flattens a pytree into a list of (path, leaf) pairs and a PyTreeDef."""
    path_leaf_pairs, internal_def = _flatten_recursive(tree, is_leaf, True, ())
    return path_leaf_pairs, PyTreeDef(internal_def)

def tree_unflatten(treedef, leaves):
    """Reconstructs a pytree from a PyTreeDef and a list of leaves."""
    if not isinstance(treedef, PyTreeDef):
        raise TypeError("tree_unflatten must be called with a PyTreeDef object.")
    return _unflatten_recursive(treedef._internal_def, iter(leaves))

def tree_map(f, tree, *rest, is_leaf=None):
    """Maps a function over the leaves of one or more pytrees."""
    leaves, treedef = tree_flatten(tree, is_leaf)
    all_leaves = [leaves]
    for other_tree in rest:
        other_leaves, other_treedef = tree_flatten(other_tree, is_leaf)
        if other_treedef != treedef:
            raise ValueError(f"treemap: pytree structure mismatch. Got {other_treedef}, expected {treedef}")
        all_leaves.append(other_leaves)
    new_leaves = [f(*xs) for xs in zip(*all_leaves)]
    return tree_unflatten(treedef, new_leaves)

def tree_map_with_path(f, tree, *rest, is_leaf=None):
    """Maps a function over the leaves of one or more pytrees, providing the path to each leaf."""
    path_leaf_pairs, treedef = tree_flatten_with_path(tree, is_leaf)
    paths, main_leaves = zip(*path_leaf_pairs) if path_leaf_pairs else ([], [])
    # Normalize paths: drop leading SequenceKey for nested paths under a root list
    normalized_paths = []
    for path in paths:
        if len(path) > 1 and isinstance(path[0], SequenceKey):
            # drop the root list index
            normalized_paths.append(path[1:])
        else:
            normalized_paths.append(path)
    paths = normalized_paths

    all_leaves = [list(main_leaves)]
    for other_tree in rest:
        other_leaves, other_treedef = tree_flatten(other_tree, is_leaf)
        if other_treedef != treedef:
            raise ValueError(f"tree_map_with_path: pytree structure mismatch. Got {other_treedef}, expected {treedef}")
        all_leaves.append(other_leaves)

    new_leaves = [f(path, *leaf_group) for path, *leaf_group in zip(paths, *all_leaves)]
    return tree_unflatten(treedef, new_leaves)

# ==============================================================================
# ================================== TESTS =====================================
# ==============================================================================
if __name__ == "__main__":
    print("--- Running Original Tests ---")
    tree1 = {'a': [1, 2, 3], 'b': {'c': 4}}
    print(type(tree_flatten(tree1)[1]))
    assert tree_map(lambda x: x * x, tree1) == {'a': [1, 4, 9], 'b': {'c': 16}}
    print("✅ Test 1 Passed")

    tree2a = [10, {'x': 20, 'y': 30}]
    tree2b = [1,  {'x': 2,  'y': 3}]
    assert tree_map(lambda x, y: x + y, tree2a, tree2b) == [11, {'x': 22, 'y': 33}]
    print("✅ Test 2 Passed")

    tree3 = {'leaves': [(1, 2), (3, 4)], 'nodes': [5, 6]}
    is_leaf_fn = lambda x: isinstance(x, tuple)
    assert tree_map(lambda x: len(x) if isinstance(x, tuple) else x + 1, tree3, is_leaf=is_leaf_fn) == {'leaves': [2, 2], 'nodes': [6, 7]}
    print("✅ Test 3 Passed")
    print("-" * 30 + "\n")

    # --- Test 4: Structure mismatch error ---
    print("--- Test 4: Structure mismatch error ---")
    tree4a = [1, 2]
    tree4b = [1, 2, 3] # Different structure (more leaves)
    try:
        tree_map(lambda x, y: x + y, tree4a, tree4b)
    except ValueError as e:
        print(f"Input 1: {tree4a}")
        print(f"Input 2: {tree4b}")
        print(f"Successfully caught expected error: {e}")
        assert "mismatch" in str(e)
    print("✅ Test 4 Passed\n")

    # --- Test 5: Flatten and Unflatten ---
    print("--- Test 5: Flatten and Unflatten ---")
    tree5 = ([1, 2], {'c': (3, 4), 'd': 5})
    leaves, treedef = tree_flatten(tree5)
    print(f"Original tree: {tree5}")
    print(f"Leaves:        {leaves}")
    print(f"Treedef:       {treedef}")
    reconstructed_tree = tree_unflatten(treedef, leaves)
    print(f"Reconstructed: {reconstructed_tree}")
    assert reconstructed_tree == tree5
    print("✅ Test 5 Passed\n")

    # --- New Tests for tree_map_with_path ---
    print("--- Test 6: Simple tree_map_with_path ---")
    path_tree = {'data': [10, 20]}
    # The mapping function receives the path as its first argument
    def add_path_str(kp, x):
        return f"{keystr(kp)}={x}"

    result6 = tree_map_with_path(add_path_str, path_tree)
    expected6 = {'data': ["['data'][0]=10", "['data'][1]=20"]}
    print(f"Input:    {path_tree}")
    print(f"Result:   {result6}")
    print(f"Expected: {expected6}")
    assert result6 == expected6
    print("✅ Test 6 Passed\n")

    print("--- Test 7: tree_map_with_path on multiple trees ---")
    path_tree_a = [1, {'val': 2}]
    path_tree_b = [10, {'val': 20}]
    
    def combine_with_path(kp, x, y):
        return (keystr(kp), x + y)

    result7 = tree_map_with_path(combine_with_path, path_tree_a, path_tree_b)
    expected7 = [('[0]', 11), {'val': ("['val']", 22)}]
    print(f"Input 1: {path_tree_a}")
    print(f"Input 2: {path_tree_b}")
    print(f"Result:   {result7}")
    print(f"Expected: {expected7}")
    assert result7 == expected7
    print("✅ Test 7 Passed\n")