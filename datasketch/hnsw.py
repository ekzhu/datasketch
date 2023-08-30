import heapq
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class HNSW(object):
    """Hierarchical Navigable Small World (HNSW) graph index for approximate
    nearest neighbor search. This implementation is based on the paper
    "Efficient and robust approximate nearest neighbor search using Hierarchical
    Navigable Small World graphs" by Yu. A. Malkov, D. A. Yashunin (2016),
    <https://arxiv.org/abs/1603.09320>`_.

    Args:
        distance_func: A function that takes two vectors and returns a float
            representing the distance between them.
        m (int): The number of neighbors to keep for each node.
        ef_construction (int): The number of neighbors to consider during
            construction.
        m0 (Optional[int]): The number of neighbors to keep for each node at
            the 0th level. If None, defaults to 2 * m.

    Example:

        .. code-block:: python

            import hnsw
            import numpy as np
            data = np.random.random_sample((1000, 10))
            index = hnsw.HNSW(distance_func=lambda x, y: np.linalg.norm(x - y), m=5, efConstruction=200)
            for i, d in enumerate(data):
                index.add(i, d)
            index.search(data[0], k=10)

    """

    def __init__(
        self,
        distance_func: Callable[[np.ndarray, np.ndarray], float],
        m: int = 5,
        ef_construction: int = 200,
        m0: Optional[int] = None,
    ) -> None:
        self._data: Dict[Any, np.ndarray] = {}
        self._distance_func = distance_func
        self._m = m
        self._efConstruction = ef_construction
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / np.log(m)
        # self._graphs[level][i] contains a {j: dist} dictionary,
        # where j is a neighbor of i and dist is distance
        self._graphs: List[Dict[Any, Dict[Any, float]]] = []
        self._entry_point = None

    def __len__(self):
        return len(self._data)

    def __contains__(self, idx):
        return idx in self._data

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return (
            f"HNSW({self._distance_func}, m={self._m}, "
            f"efConstruction={self._efConstruction}, m0={self._m0}, "
            f"num_points={len(self._data)}, num_levels={len(self._graphs)})"
        )

    def add(
        self,
        new_id: Any,
        new_point: np.ndarray,
        ef: Optional[int] = None,
        level: Optional[int] = None,
    ) -> None:
        """Add a new point to the index.

        Args:
            new_id (Any): The id of the new point.
            new_point (np.ndarray): The new point to add to the index.
            ef (Optional[int]): The number of neighbors to consider during insertion.
            level (Optional[int]): The level at which to insert the new point.

        Raises:
            ValueError: If the new_id already exists in the index.
        """
        if ef is None:
            ef = self._efConstruction
        if new_id in self._data:
            raise ValueError("Duplicate element")
        # level is the level at which we insert the element.
        if level is None:
            level = int(-np.log(np.random.random_sample()) * self._level_mult) + 1
        self._data[new_id] = new_point
        if (
            self._entry_point is not None
        ):  # The HNSW is not empty, we have an entry point
            dist = self._distance_func(new_point, self._data[self._entry_point])
            point = self._entry_point
            # For all levels in which we dont have to insert elem,
            # we search for the closest neighbor using greedy search.
            for layer in reversed(self._graphs[level:]):
                point, dist = self._search_ef1(new_point, point, dist, layer)
            # Entry points for search at each level to insert.
            entry_points = [(-dist, point)]
            for layer in reversed(self._graphs[:level]):
                # Maximum number of neighbors to keep at this level.
                level_m = self._m if layer is not self._graphs[0] else self._m0
                # Search this layer for neighbors to insert, and update entry points
                # for the next level.
                entry_points = self._search_base_layer(
                    new_point, entry_points, layer, ef
                )
                # Insert the new node into the graph with out-going edges.
                # We prune the out-going edges to keep only the top level_m neighbors.
                layer[new_id] = {
                    p: d
                    for d, p in self._heuristic_prune(
                        [(-mdist, p) for mdist, p in entry_points], level_m
                    )
                }
                # For each neighbor of the new node, we insert the new node as a neighbor.
                for neighbor_idx, dist in layer[new_id].items():
                    layer[neighbor_idx] = {
                        p: d
                        # We prune the edges to keep only the top level_m neighbors
                        # based on heuristic.
                        for d, p in self._heuristic_prune(
                            [(d, p) for p, d in layer[neighbor_idx].items()]
                            + [(dist, new_id)],
                            level_m,
                        )
                    }
        # For all levels above the current level, we create an empty graph.
        for _ in range(len(self._graphs), level):
            self._graphs.append({new_id: {}})
            # We set the entry point for each new level to be the new node.
            self._entry_point = new_id

    def search(
        self, query: np.ndarray, k: Optional[int] = None, ef: Optional[int] = None
    ) -> List[Tuple[Any, float]]:
        """Search for the k nearest neighbors of the query point.

        Args:
            query (np.ndarray): The query point.
            k (Optional[int]): The number of neighbors to return. If None, return
                all neighbors found.
            ef (Optional[int]): The number of neighbors to consider during search.
                If None, use the construction ef.

        Returns:
            List[Tuple[Any, float]]: A list of (id, distance) pairs for the k
                nearest neighbors of the query point.

        Raises:
            ValueError: If the entry point is not found.
        """
        if ef is None:
            ef = self._efConstruction
        if self._entry_point is None:
            raise ValueError("Entry point not found.")
        entry_point_dist = self._distance_func(query, self._data[self._entry_point])
        entry_point = self._entry_point
        # Search for the closest neighbor from the highest level to the 2nd
        # level using greedy search.
        for layer in reversed(self._graphs[1:]):
            entry_point, entry_point_dist = self._search_ef1(
                query, entry_point, entry_point_dist, layer
            )
        # Search for the neighbors at the base layer using ef search.
        candidates = self._search_base_layer(
            query, [(-entry_point_dist, entry_point)], self._graphs[0], ef
        )
        if k is not None:
            # If k is specified, we return the k nearest neighbors.
            candidates = heapq.nlargest(k, candidates)
        else:
            # Otherwise, we return all neighbors found.
            candidates.sort(reverse=True)
        # Return the neighbors as a list of (id, distance) pairs.
        return [(idx, -mdist) for mdist, idx in candidates]

    def _search_ef1(
        self,
        query: np.ndarray,
        entry_point: Any,
        entry_point_dist: float,
        layer: Dict[Any, Dict[Any, float]],
    ) -> Tuple[Any, float]:
        """The greedy search algorithm for finding the closest neighbor only.

        Args:
            query (np.ndarray): The query point.
            entry_point (Any): The entry point for the search.
            entry_point_dist (float): The distance from the query point to the
                entry point.
            layer (Dict[Any, Dict[Any, float]]): The graph for the layer.

        Returns:
            Tuple[Any, float]: A tuple of (id, distance) representing the closest
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
            dists = [self._distance_func(query, self._data[p]) for p in neighbors]
            for p, d in zip(neighbors, dists):
                # Update the best node if we find a closer node.
                if d < best_dist:
                    best, best_dist = p, d
                    # Add the neighbor to the heap.
                    heapq.heappush(candidates, (d, p))
        return best, best_dist

    def _search_base_layer(
        self,
        query: np.ndarray,
        entry_points: List[Tuple[float, Any]],
        layer: Dict[Any, Dict[Any, float]],
        ef: int,
    ) -> List[Tuple[float, Any]]:
        """The ef search algorithm for finding neighbors in a given layer.

        Args:
            query (np.ndarray): The query point.
            entry_points (List[Tuple[float, Any]]): A list of (-distance, idx) pairs
                representing the entry points for the search.
            layer (Dict[Any, Dict[Any, float]]): The graph for the layer.
            ef (int): The number of neighbors to consider during search.

        Returns:
            List[Tuple[float, Any]]: A heap of (-distance, idx) pairs representing
                the neighbors found.
        """
        # candidates is a heap of (distance, idx) pairs.
        candidates = [(-mdist, p) for mdist, p in entry_points]
        heapq.heapify(candidates)
        visited = set(p for _, p in entry_points)
        while candidates:
            # Pop the closest node from the heap.
            dist, curr_idx = heapq.heappop(candidates)
            # Terminate the search if the distance to the current closest node
            # is larger than the distance to the best node.
            closet_dist = -entry_points[0][0]
            if dist > closet_dist:
                break
            neighbors = [p for p in layer[curr_idx] if p not in visited]
            visited.update(neighbors)
            dists = [self._distance_func(query, self._data[p]) for p in neighbors]
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
            candidates (List[Tuple[float, Any]]): A list of (distance, idx) pairs
                representing the potential neighbors.
            max_size (int): The maximum number of neighbors to keep.

        Returns:
            List[Tuple[float, Any]]: A list of (distance, idx) pairs representing
                the pruned neighbors.
        """
        if len(candidates) < max_size:
            # If the number of entry points is smaller than max_size, we return
            # all entry points.
            return candidates
        # candidates is a heap of (distance, idx) pairs.
        heapq.heapify(candidates)
        pruned = []
        while candidates:
            if len(pruned) >= max_size:
                break
            # Pop the closest node from the heap.
            candidate_dist, candidate_idx = heapq.heappop(candidates)
            good = True
            for _, selected_idx in pruned:
                dist_to_selected_neighbor = self._distance_func(
                    self._data[selected_idx], self._data[candidate_idx]
                )
                if dist_to_selected_neighbor < candidate_dist:
                    good = False
                    break
            if good:
                pruned.append((candidate_dist, candidate_idx))
        return pruned
