import heapq
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


class _Layer(object):
    """A graph layer in the HNSW index. This is a dictionary-like object
    that maps a key to a dictionary of neighbors.

    Args:
        key (Any): The first key to insert into the graph.
    """

    def __init__(self, key: Any) -> None:
        # self._graph[key] contains a {j: dist} dictionary,
        # where j is a neighbor of key and dist is distance.
        self._graph: Dict[Any, Dict[Any, float]] = {key: {}}
        # self._reverse_edges[key] contains a set of neighbors of key.
        self._reverse_edges: Dict[Any, Set] = {}

    def __contains__(self, key: Any) -> bool:
        return key in self._graph

    def __getitem__(self, key: Any) -> Dict[Any, float]:
        return self._graph[key]

    def __setitem__(self, key: Any, value: Dict[Any, float]) -> None:
        old_neighbors = self._graph.get(key, {})
        self._graph[key] = value
        for neighbor in old_neighbors:
            self._reverse_edges[neighbor].discard(key)
        for neighbor in value:
            self._reverse_edges.setdefault(neighbor, set()).add(key)


class _Layer(object):
    """A graph layer in the HNSW index. This is a dictionary-like object
    that maps a key to a dictionary of neighbors.

    Args:
        key (Any): The first key to insert into the graph.
    """

    def __init__(self, key: Any) -> None:
        # self._graph[key] contains a {j: dist} dictionary,
        # where j is a neighbor of key and dist is distance.
        self._graph: Dict[Any, Dict[Any, float]] = {key: {}}
        # self._reverse_edges[key] contains a set of neighbors of key.
        self._reverse_edges: Dict[Any, Set] = {}

    def __contains__(self, key: Any) -> bool:
        return key in self._graph

    def __getitem__(self, key: Any) -> Dict[Any, float]:
        return self._graph[key]

    def __setitem__(self, key: Any, value: Dict[Any, float]) -> None:
        old_neighbors = self._graph.get(key, {})
        self._graph[key] = value
        for neighbor in old_neighbors:
            self._reverse_edges[neighbor].discard(key)
        for neighbor in value:
            self._reverse_edges.setdefault(neighbor, set()).add(key)


class HNSW(object):
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

    Example:

        .. code-block:: python

            import hnsw
            import numpy as np
            data = np.random.random_sample((1000, 10))
            index = hnsw.HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
            for i, d in enumerate(data):
                index.insert(i, d)
            index.query(data[0], k=10)

    """

    def __init__(
        self,
        distance_func: Callable[[np.ndarray, np.ndarray], float],
        m: int = 16,
        ef_construction: int = 200,
        m0: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._data: Dict[Any, np.ndarray] = {}
        self._distance_func = distance_func
        self._m = m
        self._ef_construction = ef_construction
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / np.log(m)
        self._graphs: List[_Layer] = []
        self._entry_point = None
        self._random = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: Any) -> bool:
        return key in self._data

    def __getitem__(self, key: Any) -> np.ndarray:
        """Get the point associated with the key."""
        return self._data[key]

    def items(self) -> Iterable[Tuple[Any, np.ndarray]]:
        """Get an iterator over (key, point) pairs in the index."""
        return self._data.items()

    def keys(self) -> Iterable[Any]:
        """Get an iterator over keys in the index."""
        return self._data.keys()

    def values(self) -> Iterable[np.ndarray]:
        """Get an iterator over points in the index."""
        return self._data.values()

    def clear(self) -> None:
        """Clear the index of all data points."""
        self._data = {}
        self._graphs = []
        self._entry_point = None

    def insert(
        self,
        key: Any,
        new_point: np.ndarray,
        ef: Optional[int] = None,
        level: Optional[int] = None,
    ) -> None:
        """Add a new point to the index.

        Args:
            key (Any): The key of the new point. If the key already exists in the
                index, the point will be updated and the index will be repaired.
            new_point (np.ndarray): The new point to add to the index.
            ef (Optional[int]): The number of neighbors to consider during insertion.
            level (Optional[int]): The level at which to insert the new point.

        """
        if ef is None:
            ef = self._ef_construction
        if key in self._data:
            self._update(key, new_point, ef)
            return
        # level is the level at which we insert the element.
        if level is None:
            level = int(-np.log(self._random.random_sample()) * self._level_mult)
        self._data[key] = new_point
        if (
            self._entry_point is not None
        ):  # The HNSW is not empty, we have an entry point
            dist = self._distance_func(new_point, self._data[self._entry_point])
            point = self._entry_point
            # For all levels in which we dont have to insert elem,
            # we search for the closest neighbor using greedy search.
            for layer in reversed(self._graphs[level + 1 :]):
                point, dist = self._search_ef1(new_point, point, dist, layer)
            # Entry points for search at each level to insert.
            entry_points = [(-dist, point)]
            for layer in reversed(self._graphs[: level + 1]):
                # Maximum number of neighbors to keep at this level.
                level_m = self._m if layer is not self._graphs[0] else self._m0
                # Search this layer for neighbors to insert, and update entry points
                # for the next level.
                entry_points = self._search_base_layer(
                    new_point, entry_points, layer, ef
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
        for _ in range(len(self._graphs), level):
            self._graphs.append(_Layer(key))
            # We set the entry point for each new level to be the new node.
            self._entry_point = key

    def _update(self, key: Any, new_point: np.ndarray, ef: int) -> None:
        """Update the point associated with the key in the index.

        Args:
            key (Any): The key of the point.
            new_point (np.ndarray): The new point to update to.
            ef (int): The number of neighbors to consider during insertion.

        Raises:
            ValueError: If the key does not exist in the index.
        """
        if key not in self._data:
            raise ValueError("Key not found in index.")
        # Update the point.
        self._data[key] = new_point
        # If the entry point is the only point in the index, we do not need to
        # update the index.
        if self._entry_point == key and len(self._data) == 1:
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
                    dist = self._distance_func(self._data[candidate_key], self._data[p])
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
        self._repair_connections_for_update(key, new_point, ef)

    def _repair_connections_for_update(
        self,
        key: Any,
        new_point: np.ndarray,
        ef: int,
    ) -> None:
        entry_point = self._entry_point
        entry_point_dist = self._distance_func(new_point, self._data[entry_point])
        entry_points = [(-entry_point_dist, entry_point)]
        for layer in reversed(self._graphs):
            if key not in layer:
                # Greedy search for the closest neighbor from the highest layer down.
                entry_point, entry_point_dist = self._search_ef1(
                    new_point, entry_point, entry_point_dist, layer
                )
                entry_points = [(-entry_point_dist, entry_point)]
            else:
                # Search for the neighbors at this layer using ef search.
                level_m = self._m if layer is not self._graphs[0] else self._m0
                entry_points = self._search_base_layer(
                    new_point, entry_points, layer, ef
                )
                # Filter out the updated node itself.
                filtered_candidates = [(-md, p) for md, p in entry_points if p != key]
                if len(filtered_candidates) == 0:
                    continue
                # Update the out-going edges of the updated node at this level.
                layer[key] = {
                    p: d for d, p in self._heuristic_prune(filtered_candidates, level_m)
                }

    def query(
        self,
        query_point: np.ndarray,
        k: Optional[int] = None,
        ef: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        """Search for the k nearest neighbors of the query point.

        Args:
            query (np.ndarray): The query point.
            k (Optional[int]): The number of neighbors to return. If None, return
                all neighbors found.
            ef (Optional[int]): The number of neighbors to consider during search.
                If None, use the construction ef.

        Returns:
            List[Tuple[Any, float]]: A list of (key, distance) pairs for the k
                nearest neighbors of the query point.

        Raises:
            ValueError: If the entry point is not found.
        """
        if ef is None:
            ef = self._ef_construction
        if self._entry_point is None:
            raise ValueError("Entry point not found.")
        entry_point_dist = self._distance_func(
            query_point, self._data[self._entry_point]
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
        entry_point: Any,
        entry_point_dist: float,
        layer: _Layer,
    ) -> Tuple[Any, float]:
        """The greedy search algorithm for finding the closest neighbor only.

        Args:
            query (np.ndarray): The query point.
            entry_point (Any): The entry point for the search.
            entry_point_dist (float): The distance from the query point to the
                entry point.
            layer (_Layer): The graph for the layer.

        Returns:
            Tuple[Any, float]: A tuple of (key, distance) representing the closest
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
            neighbors = [p for p in layer[curr] if p not in visited]
            visited.update(neighbors)
            dists = [self._distance_func(query_point, self._data[p]) for p in neighbors]
            for p, d in zip(neighbors, dists):
                # Update the best node if we find a closer node.
                if d < best_dist:
                    best, best_dist = p, d
                    # Add the neighbor to the heap.
                    heapq.heappush(candidates, (d, p))
        return best, best_dist

    def _search_base_layer(
        self,
        query_point: np.ndarray,
        entry_points: List[Tuple[float, Any]],
        layer: _Layer,
        ef: int,
    ) -> List[Tuple[float, Any]]:
        """The ef search algorithm for finding neighbors in a given layer.

        Args:
            query (np.ndarray): The query point.
            entry_points (List[Tuple[float, Any]]): A list of (-distance, key) pairs
                representing the entry points for the search.
            layer (_Layer): The graph for the layer.
            ef (int): The number of neighbors to consider during search.

        Returns:
            List[Tuple[float, Any]]: A heap of (-distance, key) pairs representing
                the neighbors found.
        """
        # candidates is a heap of (distance, key) pairs.
        candidates = [(-mdist, p) for mdist, p in entry_points]
        heapq.heapify(candidates)
        visited = set(p for _, p in entry_points)
        while candidates:
            # Pop the closest node from the heap.
            dist, curr_key = heapq.heappop(candidates)
            # Terminate the search if the distance to the current closest node
            # is larger than the distance to the best node.
            closet_dist = -entry_points[0][0]
            if dist > closet_dist:
                break
            neighbors = [p for p in layer[curr_key] if p not in visited]
            visited.update(neighbors)
            dists = [self._distance_func(query_point, self._data[p]) for p in neighbors]
            for p, dist in zip(neighbors, dists):
                if len(entry_points) < ef:
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
        self, candidates: List[Tuple[float, Any]], max_size: int
    ) -> List[Tuple[float, Any]]:
        """Prune the potential neigbors to keep only the top max_size neighbors.
        This algorithm is based on hnswlib's heuristic pruning algorithm:
        <https://github.com/nmslib/hnswlib/blob/978f7137bc9555a1b61920f05d9d0d8252ca9169/hnswlib/hnswalg.h#L382>`_.

        Args:
            candidates (List[Tuple[float, Any]]): A list of (distance, key) pairs
                representing the potential neighbors.
            max_size (int): The maximum number of neighbors to keep.

        Returns:
            List[Tuple[float, Any]]: A list of (distance, key) pairs representing
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
                    self._data[selected_key], self._data[candidate_key]
                )
                if dist_to_selected_neighbor < candidate_dist:
                    good = False
                    break
            if good:
                pruned.append((candidate_dist, candidate_key))
        return pruned
