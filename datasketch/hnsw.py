from __future__ import annotations
from collections import OrderedDict
import heapq
from typing import (
    Hashable,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np


class _Layer(object):
    """A graph layer in the HNSW index. This is a dictionary-like object
    that maps a key to a dictionary of neighbors.

    Args:
        key (Hashable): The first key to insert into the graph.
    """

    def __init__(self, key: Hashable) -> None:
        # self._graph[key] contains a {j: dist} dictionary,
        # where j is a neighbor of key and dist is distance.
        self._graph: Dict[Hashable, Dict[Hashable, float]] = {key: {}}

    def __contains__(self, key: Hashable) -> bool:
        return key in self._graph

    def __getitem__(self, key: Hashable) -> Dict[Hashable, float]:
        return self._graph[key]

    def __setitem__(self, key: Hashable, value: Dict[Hashable, float]) -> None:
        self._graph[key] = value

    def __delitem__(self, key: Hashable) -> None:
        del self._graph[key]

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, _Layer):
            return False
        return self._graph == __value._graph

    def __len__(self) -> int:
        return len(self._graph)

    def __iter__(self) -> Iterable[Hashable]:
        return iter(self._graph)

    def copy(self) -> _Layer:
        """Create a copy of the layer."""
        new_layer = _Layer(None)
        new_layer._graph = {k: dict(v) for k, v in self._graph.items()}
        return new_layer

    def get_reverse_edges(self, key: Hashable) -> Set[Hashable]:
        reverse_edges = set()
        for neighbor, neighbors in self._graph.items():
            if key in neighbors:
                reverse_edges.add(neighbor)
        return reverse_edges


class _LayerWithReversedEdges(_Layer):
    """A graph layer in the HNSW index that also maintains reverse edges.

    Args:
        key (Hashable): The first key to insert into the graph.
    """

    def __init__(self, key: Hashable) -> None:
        # self._graph[key] contains a {j: dist} dictionary,
        # where j is a neighbor of key and dist is distance.
        self._graph: Dict[Hashable, Dict[Hashable, float]] = {key: {}}
        # self._reverse_edges[key] contains a set of neighbors of key.
        self._reverse_edges: Dict[Hashable, Set] = {}

    def __setitem__(self, key: Hashable, value: Dict[Hashable, float]) -> None:
        old_neighbors = self._graph.get(key, {})
        self._graph[key] = value
        for neighbor in old_neighbors:
            self._reverse_edges[neighbor].discard(key)
        for neighbor in value:
            self._reverse_edges.setdefault(neighbor, set()).add(key)
        if key not in self._reverse_edges:
            self._reverse_edges[key] = set()

    def __delitem__(self, key: Hashable) -> None:
        old_neighbors = self._graph.get(key, {})
        del self._graph[key]
        for neighbor in old_neighbors:
            self._reverse_edges[neighbor].discard(key)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, _LayerWithReversedEdges):
            return False
        return (
            self._graph == __value._graph
            and self._reverse_edges == __value._reverse_edges
        )

    def __len__(self) -> int:
        return len(self._graph)

    def __iter__(self) -> Iterable[Hashable]:
        return iter(self._graph)

    def copy(self) -> _LayerWithReversedEdges:
        """Create a copy of the layer."""
        new_layer = _LayerWithReversedEdges(None)
        new_layer._graph = {k: dict(v) for k, v in self._graph.items()}
        new_layer._reverse_edges = {k: set(v) for k, v in self._reverse_edges.items()}
        return new_layer

    def get_reverse_edges(self, key: Hashable) -> Set[Hashable]:
        return self._reverse_edges[key]


class _Node(object):
    """A node in the HNSW graph."""

    def __init__(self, key: Hashable, point: np.ndarray, is_deleted=False) -> None:
        self.key = key
        self.point = point
        self.is_deleted = is_deleted

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, _Node):
            return False
        return (
            self.key == __value.key
            and np.array_equal(self.point, __value.point)
            and self.is_deleted == __value.is_deleted
        )

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return (
            f"_Node(key={self.key}, point={self.point}, is_deleted={self.is_deleted})"
        )

    def copy(self) -> _Node:
        return _Node(self.key, self.point, self.is_deleted)


class HNSW(MutableMapping):
    """Hierarchical Navigable Small World (HNSW) graph index for approximate
    nearest neighbor search. This implementation is based on the paper
    "Efficient and robust approximate nearest neighbor search using Hierarchical
    Navigable Small World graphs" by Yu. A. Malkov, D. A. Yashunin (2016),
    `<https://arxiv.org/abs/1603.09320>`_.

    Args:
        distance_func: A function that takes two vectors and returns a float
            representing the distance between them.
        m (int): The number of neighbors to keep for each node.
        ef_construction (int): The number of neighbors to consider during
            construction.
        m0 (Optional[int]): The number of neighbors to keep for each node at
            the 0th level. If None, defaults to 2 * m.
        seed (Optional[int]): The random seed to use for the random number
            generator.
        reverse_edges (bool): Whether to maintain reverse edges in the graph.
            This speeds up hard remove (:meth:`remove`) but increases memory
            usage and slows down :meth:`insert`.

    Examples:

        Create an HNSW index with Euclidean distance and insert 1000 random
        vectors of dimension 10.

        .. code-block:: python

            from datasketch.hnsw import HNSW
            import numpy as np

            data = np.random.random_sample((1000, 10))
            index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
            for i, d in enumerate(data):
                index.insert(i, d)

            # Query the index for the 10 nearest neighbors of the first vector.
            index.query(data[0], k=10)

        Create an HNSW index with Jaccard distance and insert 1000 random
        sets of 10 elements each.

        .. code-block:: python

            from datasketch.hnsw import HNSW
            import numpy as np

            # Each set is represented as a 10-element vector of random integers
            # between 0 and 100.
            # Deduplication is handled by the distance function.
            data = np.random.randint(0, 100, size=(1000, 10))
            jaccard_distance = lambda x, y: (
                1.0 - float(len(np.intersect1d(x, y, assume_unique=False)))
                / float(len(np.union1d(x, y)))
            )
            index = HNSW(distance_func=jaccard_distance)
            for i, d in enumerate(data):
                index[i] = d

            # Query the index for the 10 nearest neighbors of the first set.
            index.query(data[0], k=10)

    """

    def __init__(
        self,
        distance_func: Callable[[np.ndarray, np.ndarray], float],
        m: int = 16,
        ef_construction: int = 200,
        m0: Optional[int] = None,
        seed: Optional[int] = None,
        reversed_edges: bool = False,
    ) -> None:
        self._nodes: OrderedDict[Hashable, _Node] = OrderedDict()
        self._distance_func = distance_func
        self._m = m
        self._ef_construction = ef_construction
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / np.log(m)
        self._graphs: List[_Layer] = []
        self._entry_point = None
        self._random = np.random.RandomState(seed)
        self._layer_class = _LayerWithReversedEdges if reversed_edges else _Layer

    def __len__(self) -> int:
        """Return the number of points in the index excluding those
        that were soft-removed."""
        return sum(not node.is_deleted for node in self._nodes.values())

    def __contains__(self, key: Hashable) -> bool:
        """Return ``True`` if the index contains the key and it was
        not soft-removed, else ``False``."""
        return key in self._nodes and not self._nodes[key].is_deleted

    def __getitem__(self, key: Hashable) -> np.ndarray:
        """Get the point associated with the key. Raises KeyError if the key
        does not exist in the index or it was soft-removed."""
        if key not in self:
            raise KeyError(key)
        return self._nodes[key].point

    def __setitem__(self, key: Hashable, value: np.ndarray) -> None:
        """Set the point associated with the key and update the index.
        This is equivalent to calling :meth:`insert` with the key and point."""
        self.insert(key, value)

    def __delitem__(self, key: Hashable) -> None:
        """Soft remove the point associated with the key. Raises a KeyError if the
        key does not exist in the index. This is equivalent to calling
        :meth:`remove` with the key.
        """
        self.remove(key)

    def __iter__(self) -> Iterator[Hashable]:
        """Return an iterator over the keys of the index that were not
        soft-removed."""
        return (key for key in self._nodes if not self._nodes[key].is_deleted)

    def reversed(self) -> Iterator[Hashable]:
        """Return a reverse iterator over the keys of the index that were not
        soft-removed."""
        return (key for key in reversed(self._nodes) if not self._nodes[key].is_deleted)

    def __eq__(self, __value: object) -> bool:
        """Return True only if the index parameters, random states, keys, points
        points, and index structures are equal, including deleted points."""
        if not isinstance(__value, HNSW):
            return False
        # Check if the index parameters are equal.
        if (
            self._distance_func != __value._distance_func
            or self._m != __value._m
            or self._ef_construction != __value._ef_construction
            or self._m0 != __value._m0
            or self._level_mult != __value._level_mult
            or self._entry_point != __value._entry_point
        ):
            return False
        # Check if the random states are equal.
        rand_state_1 = self._random.get_state()
        rand_state_2 = __value._random.get_state()
        for i in range(len(rand_state_1)):
            if isinstance(rand_state_1[i], np.ndarray):
                if not np.array_equal(rand_state_1[i], rand_state_2[i]):
                    return False
            else:
                if rand_state_1[i] != rand_state_2[i]:
                    return False
        # Check if keys and points are equal.
        # Note that deleted points are compared too.
        return (
            all(key in self._nodes for key in __value._nodes)
            and all(key in __value._nodes for key in self._nodes)
            and all(self._nodes[key] == __value._nodes[key] for key in self._nodes)
            and self._graphs == __value._graphs
        )

    def get(
        self, key: Hashable, default: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, None]:
        """Return the point for key in the index, else default. If default is not
        given and key is not in the index or it was soft-removed, return None."""
        if key not in self:
            return default
        return self._nodes[key].point

    def items(self) -> Iterator[Tuple[Hashable, np.ndarray]]:
        """Return an iterator of the indexed points that were not soft-removed
        as (key, point) pairs."""
        return (
            (key, node.point)
            for key, node in self._nodes.items()
            if not node.is_deleted
        )

    def keys(self) -> Iterator[Hashable]:
        """Return an iterator of the keys of the index points that were not
        soft-removed."""
        return (key for key in self._nodes if not self._nodes[key].is_deleted)

    def values(self) -> Iterator[np.ndarray]:
        """Return an iterator of the index points that were not soft-removed."""
        return (node.point for node in self._nodes.values() if not node.is_deleted)

    def pop(
        self, key: Hashable, default: Optional[np.ndarray] = None, hard: bool = False
    ) -> np.ndarray:
        """If key is in the index, remove it and return its associated point,
        else return default. If default is not given and key is not in the index
        or it was soft-removed, raise KeyError.
        """
        if key not in self:
            if default is None:
                raise KeyError(key)
            return default
        point = self._nodes[key].point
        self.remove(key, hard=hard)
        return point

    def popitem(
        self, last: bool = True, hard: bool = False
    ) -> Tuple[Hashable, np.ndarray]:
        """Remove and return a (key, point) pair from the index. Pairs are
        returned in LIFO order if `last` is true or FIFO order if false.
        If the index is empty or all points are soft-removed, raise KeyError.

        Note:
            In versions of Python before 3.7, the order of items in the index
            is not guaranteed. This method will remove and return an arbitrary
            (key, point) pair.
        """
        if not self._nodes:
            raise KeyError("popitem(): index is empty")
        if last:
            key = next(
                (
                    key
                    for key in reversed(self._nodes)
                    if not self._nodes[key].is_deleted
                ),
                None,
            )
        else:
            key = next(
                (key for key in self._nodes if not self._nodes[key].is_deleted), None
            )
        if key is None:
            raise KeyError("popitem(): index is empty")
        point = self._nodes[key].point
        self.remove(key, hard=hard)
        return key, point

    def clear(self) -> None:
        """Clear the index of all data points. This will not reset the random
        number generator."""
        self._nodes = {}
        self._graphs = []
        self._entry_point = None

    def copy(self) -> HNSW:
        """Create a copy of the index. The copy will have the same parameters
        as the original index and the same keys and points, but will not share
        any index data structures (i.e., graphs) with the original index.
        The new index's random state will start from a copy of the original
        index's."""
        new_index = HNSW(
            self._distance_func,
            m=self._m,
            ef_construction=self._ef_construction,
            m0=self._m0,
        )
        new_index._nodes = OrderedDict(
            (key, node.copy()) for key, node in self._nodes.items()
        )
        new_index._graphs = [layer.copy() for layer in self._graphs]
        new_index._entry_point = self._entry_point
        new_index._random.set_state(self._random.get_state())
        return new_index

    def update(self, other: Union[Mapping, HNSW]) -> None:
        """Update the index with the points from the other Mapping or HNSW object,
        overwriting existing keys.

        Args:
            other (Union[Mapping, HNSW]): The other Mapping or HNSW object.

        Examples:

            Create an HNSW index with a dictionary of points.

            .. code-block:: python

                from datasketch.hnsw import HNSW
                import numpy as np

                data = np.random.random_sample((1000, 10))
                index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))

                # Batch insert 1000 points.
                index.update({i: d for i, d in enumerate(data)})

            Create an HNSW index with another HNSW index.

            .. code-block:: python

                from datasketch.hnsw import HNSW
                import numpy as np

                data = np.random.random_sample((1000, 10))
                index1 = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
                index2 = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))

                # Batch insert 1000 points.
                index1.update({i: d for i, d in enumerate(data)})

                # Update index2 with the points from index1.
                index2.update(index1)

        """
        for key, point in other.items():
            self.insert(key, point)

    def setdefault(self, key: Hashable, default: np.ndarray) -> np.ndarray:
        """If key is in the index and it was not soft-removed, return
        its associated point. If not, insert
        key with a value of default and return default. default cannot be None."""
        if default is None:
            raise ValueError("Default value cannot be None.")
        if key not in self._nodes or self._nodes[key].is_deleted:
            self.insert(key, default)
        return self._nodes[key]

    def insert(
        self,
        key: Hashable,
        new_point: np.ndarray,
        ef: Optional[int] = None,
        level: Optional[int] = None,
    ) -> None:
        """Add a new point to the index.

        Args:
            key (Hashable): The key of the new point. If the key already exists in the
                index, the point will be updated and the index will be repaired.
            new_point (np.ndarray): The new point to add to the index.
            ef (Optional[int]): The number of neighbors to consider during insertion.
                If None, use the construction ef.
            level (Optional[int]): The level at which to insert the new point.
                If None, the level will be chosen automatically.

        """
        if ef is None:
            ef = self._ef_construction
        if key in self._nodes:
            if self._nodes[key].is_deleted:
                self._nodes[key].is_deleted = False
            self._update(key, new_point, ef)
            return
        # level is the level at which we insert the element.
        if level is None:
            level = int(-np.log(self._random.random_sample()) * self._level_mult)
        self._nodes[key] = _Node(key, new_point)
        if (
            self._entry_point is not None
        ):  # The HNSW is not empty, we have an entry point
            dist = self._distance_func(new_point, self._nodes[self._entry_point].point)
            point = self._entry_point
            # For all levels in which we dont have to insert elem,
            # we search for the closest neighbor using greedy search.
            for layer in reversed(self._graphs[level + 1 :]):
                point, dist = self._search_ef1(
                    new_point, point, dist, layer, allow_soft_deleted=True
                )
            # Entry points for search at each level to insert.
            entry_points = [(-dist, point)]
            for layer in reversed(self._graphs[: level + 1]):
                # Maximum number of neighbors to keep at this level.
                level_m = self._m if layer is not self._graphs[0] else self._m0
                # Search this layer for neighbors to insert, and update entry points
                # for the next level.
                entry_points = self._search_base_layer(
                    new_point, entry_points, layer, ef, allow_soft_deleted=True
                )
                # Insert the new node into the graph with out-going edges.
                # We prune the out-going edges to keep only the top level_m neighbors.
                layer[key] = {
                    p: d
                    for d, p in self._heuristic_prune(
                        [(-mdist, p) for mdist, p in entry_points], level_m
                    )
                }
                # For each neighbor of the new node, we insert the new node as a neighbor.
                for neighbor_key, dist in layer[key].items():
                    layer[neighbor_key] = {
                        p: d
                        # We prune the edges to keep only the top level_m neighbors
                        # based on heuristic.
                        for d, p in self._heuristic_prune(
                            [(d, p) for p, d in layer[neighbor_key].items()]
                            + [(dist, key)],
                            level_m,
                        )
                    }
        # For all levels above the current level, we create an empty graph.
        for _ in range(len(self._graphs), level + 1):
            self._graphs.append(self._layer_class(key))
            # We set the entry point for each new level to be the new node.
            self._entry_point = key

    def _update(self, key: Hashable, new_point: np.ndarray, ef: int) -> None:
        """Update the point associated with the key in the index.

        Args:
            key (Hashable): The key of the point.
            new_point (np.ndarray): The new point to update to.
            ef (int): The number of neighbors to consider during insertion.

        Raises:
            KeyError: If the key does not exist in the index.
        """
        if key not in self._nodes:
            raise KeyError(key)
        # Update the point.
        self._nodes[key].point = new_point
        # If the entry point is the only point in the index, we do not need to
        # update the index.
        if self._entry_point == key and len(self._nodes) == 1:
            return
        for layer in self._graphs:
            if key not in layer:
                break
            layer_m = self._m if layer is not self._graphs[0] else self._m0
            # Create a set of points in the 2nd-degree neighborhood of the key.
            neighborhood_keys = set([key])
            for p in layer[key].keys():
                neighborhood_keys.add(p)
                for p2 in layer[p].keys():
                    neighborhood_keys.add(p2)
            for p in layer[key].keys():
                # For each neighbor of the key, we connects it with the top ef
                # neighbors in the 2nd-degree neighborhood of the key.
                cands = []
                elem_to_keep = min(ef, len(neighborhood_keys) - 1)
                for candidate_key in neighborhood_keys:
                    if candidate_key == p:
                        continue
                    dist = self._distance_func(
                        self._nodes[candidate_key].point, self._nodes[p].point
                    )
                    if len(cands) < elem_to_keep:
                        heapq.heappush(cands, (-dist, candidate_key))
                    elif dist < -cands[0][0]:
                        heapq.heappushpop(cands, (-dist, candidate_key))
                layer[p] = {
                    p2: d2
                    for d2, p2 in self._heuristic_prune(
                        [(-md, p) for md, p in cands], layer_m
                    )
                }
        self._repair_connections(key, new_point, ef)

    def _repair_connections(
        self,
        key: Hashable,
        new_point: np.ndarray,
        ef: int,
        key_to_delete: Optional[Hashable] = None,
    ) -> None:
        entry_point = self._entry_point
        entry_point_dist = self._distance_func(
            new_point, self._nodes[entry_point].point
        )
        entry_points = [(-entry_point_dist, entry_point)]
        for layer in reversed(self._graphs):
            if key not in layer:
                # Greedy search for the closest neighbor from the highest layer down.
                entry_point, entry_point_dist = self._search_ef1(
                    new_point,
                    entry_point,
                    entry_point_dist,
                    layer,
                    # We allow soft-deleted points to be returned and used as entry point.
                    allow_soft_deleted=True,
                    key_to_hard_delete=key_to_delete,
                )
                entry_points = [(-entry_point_dist, entry_point)]
            else:
                # Search for the neighbors at this layer using ef search.
                level_m = self._m if layer is not self._graphs[0] else self._m0
                entry_points = self._search_base_layer(
                    new_point,
                    entry_points,
                    layer,
                    ef + 1,  # We add 1 to ef to account for the point itself.
                    # We allow soft-deleted points to be returned and used as entry point
                    # and neighbor candidates.
                    allow_soft_deleted=True,
                    key_to_hard_delete=key_to_delete,
                )
                # Filter out the updated node itself.
                filtered_candidates = [(-md, p) for md, p in entry_points if p != key]
                # Update the out-going edges of the updated node at this level.
                layer[key] = {
                    p: d for d, p in self._heuristic_prune(filtered_candidates, level_m)
                }

    def query(
        self,
        query_point: np.ndarray,
        k: Optional[int] = None,
        ef: Optional[int] = None,
    ) -> List[Tuple[Hashable, float]]:
        """Search for the k nearest neighbors of the query point.

        Args:
            query_point (np.ndarray): The query point.
            k (Optional[int]): The number of neighbors to return. If None, return
                all neighbors found.
            ef (Optional[int]): The number of neighbors to consider during search.
                If None, use the construction ef.

        Returns:
            List[Tuple[Hashable, float]]: A list of (key, distance) pairs for the k
                nearest neighbors of the query point.

        Raises:
            ValueError: If the entry point is not found.
        """
        if ef is None:
            ef = self._ef_construction
        if self._entry_point is None:
            raise ValueError("Entry point not found.")
        entry_point_dist = self._distance_func(
            query_point, self._nodes[self._entry_point].point
        )
        entry_point = self._entry_point
        # Search for the closest neighbor from the highest level to the 2nd
        # level using greedy search.
        for layer in reversed(self._graphs[1:]):
            entry_point, entry_point_dist = self._search_ef1(
                query_point, entry_point, entry_point_dist, layer
            )
        # Search for the neighbors at the base layer using ef search.
        candidates = self._search_base_layer(
            query_point, [(-entry_point_dist, entry_point)], self._graphs[0], ef
        )
        if k is not None:
            # If k is specified, we return the k nearest neighbors.
            candidates = heapq.nlargest(k, candidates)
        else:
            # Otherwise, we return all neighbors found.
            candidates.sort(reverse=True)
        # Return the neighbors as a list of (id, distance) pairs.
        return [(key, -mdist) for mdist, key in candidates]

    def _search_ef1(
        self,
        query_point: np.ndarray,
        entry_point: Hashable,
        entry_point_dist: float,
        layer: _Layer,
        allow_soft_deleted: bool = False,
        key_to_hard_delete: Optional[Hashable] = None,
    ) -> Tuple[Hashable, float]:
        """The greedy search algorithm for finding the closest neighbor only.

        Args:
            query (np.ndarray): The query point.
            entry_point (Hashable): The entry point for the search.
            entry_point_dist (float): The distance from the query point to the
                entry point.
            layer (_Layer): The graph for the layer.
            allow_soft_deleted (bool): Whether to allow soft-deleted points to
                be returned.
            key_to_hard_delete (Optional[Hashable]): The key of the point to be
                hard-deleted, if any. This point should never be returned.

        Returns:
            Tuple[Hashable, float]: A tuple of (key, distance) representing the closest
                neighbor found.
        """
        candidates = [(entry_point_dist, entry_point)]
        visited = set([entry_point])
        best = entry_point
        best_dist = entry_point_dist
        while candidates:
            # Pop the closest node from the heap.
            dist, curr = heapq.heappop(candidates)
            if dist > best_dist:
                # Terminate the search if the distance to the current closest node
                # is larger than the distance to the best node.
                break
            # Find the neighbors of the current node
            neighbors = [p for p in layer[curr] if p not in visited]
            visited.update(neighbors)
            dists = [
                self._distance_func(query_point, self._nodes[p].point)
                for p in neighbors
            ]
            for p, d in zip(neighbors, dists):
                # Update the best node if we find a closer node.
                if d < best_dist:
                    if (not allow_soft_deleted and self._nodes[p].is_deleted) or (
                        p == key_to_hard_delete
                    ):
                        # If the neighbor has been deleted or to be hard-deleted,
                        # we don't update the best node but we continue to
                        # explore the neighbor's neighbors.
                        pass
                    else:
                        best, best_dist = p, d
                    # Add the neighbor to the heap.
                    heapq.heappush(candidates, (d, p))
        return best, best_dist

    def _search_base_layer(
        self,
        query_point: np.ndarray,
        entry_points: List[Tuple[float, Hashable]],
        layer: _Layer,
        ef: int,
        allow_soft_deleted: bool = False,
        key_to_hard_delete: Optional[Hashable] = None,
    ) -> List[Tuple[float, Hashable]]:
        """The ef search algorithm for finding neighbors in a given layer.

        Args:
            query (np.ndarray): The query point.
            entry_points (List[Tuple[float, Hashable]]): A list of (-distance, key) pairs
                representing the entry points for the search.
            layer (_Layer): The graph for the layer.
            ef (int): The number of neighbors to consider during search.
            allow_soft_deleted (bool): Whether to allow soft-deleted points to
                be returned.
            key_to_hard_delete (Optional[Hashable]): The key of the point to be
                hard-deleted, if any. This point should never be returned.

        Returns:
            List[Tuple[float, Hashable]]: A heap of (-distance, key) pairs representing
                the neighbors found.

        Note:
            When used together with :meth:`_search_ef1`, the input entry_points
            may contain soft-deleted points depending on the flag used in
            :meth:`_search_ef1`. Therefore, the output entry_points may contain
            soft-deleted points even if allow_soft_deleted is False. Therefore,
            the caller should check input entry_points for soft-deleted
            points if necessary.
        """
        # candidates is a heap of (distance, key) pairs.
        candidates = [(-mdist, p) for mdist, p in entry_points]
        heapq.heapify(candidates)

        visited = set(p for _, p in entry_points)
        while candidates:
            # Pop the closest node from the heap.
            dist, curr_key = heapq.heappop(candidates)

            # If the neighbor has been marked as deleted, we ,

            # Terminate the search if the distance to the current closest node
            # is larger than the distance to the best node.
            closet_dist = -entry_points[0][0]
            if dist > closet_dist:
                break
            # Find the neighbors of the current node
            neighbors = [p for p in layer[curr_key] if p not in visited]
            visited.update(neighbors)
            dists = [
                self._distance_func(query_point, self._nodes[p].point)
                for p in neighbors
            ]
            for p, dist in zip(neighbors, dists):
                if (not allow_soft_deleted and self._nodes[p].is_deleted) or (
                    p == key_to_hard_delete
                ):
                    if dist <= closet_dist:
                        # If the neighbor has been deleted or to be deleted,
                        # we add it to the heap to explore its neighbors but
                        # do not add it to the entry points.
                        heapq.heappush(candidates, (dist, p))
                elif len(entry_points) < ef:
                    heapq.heappush(candidates, (dist, p))
                    # If we have not found enough neighbors, we add the neighbor
                    # to the heap.
                    heapq.heappush(entry_points, (-dist, p))
                    closet_dist = -entry_points[0][0]
                elif dist <= closet_dist:
                    heapq.heappush(candidates, (dist, p))
                    # If we have found enough neighbors, we replace the worst
                    # neighbor with the neighbor if the neighbor is closer.
                    heapq.heapreplace(entry_points, (-dist, p))
                    closet_dist = -entry_points[0][0]

        return entry_points

    def _heuristic_prune(
        self, candidates: List[Tuple[float, Hashable]], max_size: int
    ) -> List[Tuple[float, Hashable]]:
        """Prune the potential neigbors to keep only the top max_size neighbors.
        This algorithm is based on hnswlib's heuristic pruning algorithm:
        <https://github.com/nmslib/hnswlib/blob/978f7137bc9555a1b61920f05d9d0d8252ca9169/hnswlib/hnswalg.h#L382>`_.

        Args:
            candidates (List[Tuple[float, Hashable]]): A list of (distance, key) pairs
                representing the potential neighbors.
            max_size (int): The maximum number of neighbors to keep.

        Returns:
            List[Tuple[float, Hashable]]: A list of (distance, key) pairs representing
                the pruned neighbors.
        """
        if len(candidates) < max_size:
            # If the number of entry points is smaller than max_size, we return
            # all entry points.
            return candidates
        # candidates is a heap of (distance, key) pairs.
        heapq.heapify(candidates)
        pruned = []
        while candidates:
            if len(pruned) >= max_size:
                break
            # Pop the closest node from the heap.
            candidate_dist, candidate_key = heapq.heappop(candidates)
            good = True
            for _, selected_key in pruned:
                dist_to_selected_neighbor = self._distance_func(
                    self._nodes[selected_key].point, self._nodes[candidate_key].point
                )
                if dist_to_selected_neighbor < candidate_dist:
                    good = False
                    break
            if good:
                pruned.append((candidate_dist, candidate_key))
        return pruned

    def remove(
        self,
        key: Hashable,
        hard: bool = False,
        ef: Optional[int] = None,
    ) -> None:
        """Remove a point from the index. This removal algorithm is based on
        the discussion on `hnswlib issue #4`_. There are two versions:

        * *soft remove*: the point is marked as removed from the index, but its
          data and out-going edges are kept. Future queries will not return
          the point and no new edge will direct to this point,
          but the point will still be used for graph traversal.
          This is the default behavior.
        * *hard remove*: the point is removed from the index and its data and
          out-going edges are also removed. Points with out-going edges pointing
          to the deleted point will have their out-going edges
          re-assigned using the same pruning algorithm as :meth:`insert` during
          point update.

        In both versions, if the deleted point is the current entry point,
        the entry point will be re-assigned to the next point in the highest
        layer that has other points beside the current entry point.

        Subsequent soft removes without a hard remove of the same point will
        not affect the index, **unless the point was the only point in the index
        as removing it clears the index**. This is different from :meth:`pop`
        which will always raise a KeyError if the key was removed.

        Subsequent hard removes of the same point will
        raise a KeyError. If the point is soft removed and then hard removed,
        the point will be removed from the index.
        Use :meth:`clean` for removing all soft removed points from the index.

        Args:
            key (Hashable): The key of the point to remove.
            hard (bool): If True, perform a hard remove. Otherwise, perform a
                soft remove.
            ef (Optional[int]): The number of neighbors to consider during
                re-assignment. If None, use the construction ef. This argument
                is only used when hard is True.

        Raises:
            KeyError: If the index is empty or the key does not exist in the
                index and was not soft removed.

        Example:

            .. code-block:: python

                from datasketch.hnsw import HNSW
                import numpy as np

                data = np.random.random_sample((1000, 10))
                index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
                index.update({i: d for i, d in enumerate(data)})

                # Soft remove a point with key = 0.
                index.remove(0)

                # Soft remove the same point again will not change the index
                # because the index is not empty.
                index.remove(0)

                print(0 in index)  # False

                # Hard remove the point.
                index.remove(0, hard=True)

                # Hard remove the same point again will raise a KeyError.
                # index.remove(0, hard=True)

                # Soft remove rest of the points from the index.
                for i in range(1, 1000):
                    index.remove(i)

                print(len(index))  # 0

                # Clean the index to hard remove all soft removed points.
                index.clean()

        .. _hnswlib issue #4: https://github.com/nmslib/hnswlib/issues/4
        """
        if not self._nodes or key not in self._nodes:
            raise KeyError(key)
        if self._entry_point == key:
            # If the point is the entry point, we re-assign the entry point
            # to the next point in the highest layer besides the point to be
            # deleted.
            new_entry_point = None
            for layer in reversed(list(self._graphs)):
                new_entry_point = next(
                    (p for p in layer if p != key and not self._nodes[p].is_deleted),
                    None,
                )
                if new_entry_point is not None:
                    break
                else:
                    # As the layer is going to be empty after deletion, we remove it.
                    self._graphs.pop()
            if new_entry_point is None:
                # If the point to be deleted is the only point in the index,
                # we clear the index.
                self.clear()
                return
            # Update the entry point.
            self._entry_point = new_entry_point
        if ef is None:
            ef = self._ef_construction

        # Soft remove.
        self._nodes[key].is_deleted = True
        if not hard:
            return

        # Hard remove.
        # Find all points that have out-going edges pointing to the deleted point.
        keys_to_update = set()
        for layer in self._graphs:
            if key not in layer:
                break
            keys_to_update.update(layer.get_reverse_edges(key))
        # Re-assign edges for these points.
        for key_to_update in keys_to_update:
            self._repair_connections(
                key_to_update,
                self._nodes[key_to_update].point,
                ef,
                key_to_delete=key,
            )
        # Remove the point to be deleted from the grpah.
        for layer in self._graphs:
            if key not in layer:
                break
            del layer[key]
        # Remove the point from the index.
        del self._nodes[key]

    def clean(self, ef: Optional[int] = None) -> None:
        """Remove all soft removed points from the index.

        Args:
            ef (Optional[int]): The number of neighbors to consider during
                re-assignment. If None, use the construction ef.
        """
        keys_to_remove = list(key for key in self._nodes if self._nodes[key].is_deleted)
        for key in keys_to_remove:
            self.remove(key, ef=ef, hard=True)

    def merge(self, other: HNSW) -> HNSW:
        """Create a new index by merging the current index with another index.
        The new index will contain all points from both indexes.
        If a point exists in both, the point from the other index will be used.
        The new index will have the same parameters as the current index and
        a copy of the current index's random state.

        Args:
            other (HNSW): The other index to merge with.

        Returns:
            HNSW: A new index containing all points from both indexes.

        Example:

            .. code-block:: python

                from datasketch.hnsw import HNSW
                import numpy as np

                data1 = np.random.random_sample((1000, 10))
                data2 = np.random.random_sample((1000, 10))
                index1 = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
                index2 = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))

                # Batch insert data into the indexes.
                index1.update({i: d for i, d in enumerate(data1)})
                index2.update({i + len(data1): d for i, d in enumerate(data2)})

                # Merge the indexes.
                index = index1.merge(index2)

        """
        new_index = self.copy()
        new_index.update(other)
        return new_index
