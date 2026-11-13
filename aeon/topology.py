"""
Network topologies and K-shortest-path routing.

Each undirected edge
(u, v, km) becomes two directed links with independent spectrum, so a path is a
sequence of directed link indices.
"""
import heapq
from typing import List, Dict, Tuple, Optional, Set

def _nsfnet_edges():
    """NSFNET: 14 nodes, 21 bidirectional links with realistic km distances."""
    return [
        (0, 1, 1100),
        (0, 2, 1600),
        (0, 7, 2800),
        (1, 2, 600),
        (1, 3, 1000),
        (2, 5, 2000),
        (3, 4, 600),
        (3, 10, 2400),
        (4, 5, 1100),
        (4, 6, 800),
        (5, 9, 1200),
        (5, 13, 2000),
        (6, 7, 700),
        (7, 8, 700),
        (8, 9, 900),
        (8, 11, 500),
        (8, 13, 300),
        (9, 12, 500),
        (10, 11, 800),
        (10, 12, 500),
        (11, 13, 300),
    ]


def _usnet_edges():
    """USNET: 24 nodes, 43 bidirectional links with realistic km distances."""
    return [
        (0, 1, 350),
        (0, 2, 710),
        (0, 7, 530),
        (1, 2, 400),
        (1, 3, 380),
        (2, 5, 360),
        (3, 4, 610),
        (3, 10, 1090),
        (4, 5, 350),
        (4, 6, 480),
        (5, 9, 520),
        (5, 19, 900),
        (6, 7, 320),
        (6, 8, 740),
        (7, 11, 390),
        (8, 9, 460),
        (8, 12, 510),
        (9, 13, 390),
        (10, 11, 360),
        (10, 14, 440),
        (11, 15, 490),
        (12, 13, 340),
        (12, 16, 560),
        (13, 17, 420),
        (14, 15, 310),
        (14, 18, 510),
        (15, 16, 460),
        (15, 19, 680),
        (16, 17, 370),
        (16, 20, 550),
        (17, 21, 470),
        (18, 19, 340),
        (18, 22, 430),
        (19, 20, 310),
        (19, 23, 610),
        (20, 21, 380),
        (20, 22, 590),
        (21, 23, 440),
        (22, 23, 360),
        (1, 8, 890),
        (2, 4, 390),
        (7, 10, 620),
        (13, 21, 820),
    ]


def _european_edges():
    """PAN-EUROPEAN topology: 28 nodes, 41 bidirectional links with inter-node
    span lengths in km. The length of the 21-24 span (~480 km) is an estimate."""
    return [
        (0, 2, 834), (0, 5, 760), (1, 3, 462), (1, 4, 690), (2, 6, 747),
        (3, 9, 1067), (4, 6, 514), (4, 9, 540), (5, 7, 796), (6, 7, 594),
        (6, 8, 393), (6, 11, 600), (7, 10, 507), (8, 9, 259), (8, 13, 474),
        (9, 12, 552), (10, 11, 218), (10, 14, 327), (11, 13, 271), (12, 13, 592),
        (12, 16, 381), (13, 15, 456), (14, 15, 522), (14, 19, 720), (15, 16, 757),
        (15, 20, 534), (16, 17, 540), (16, 21, 420), (16, 25, 775), (17, 18, 722),
        (18, 23, 623), (19, 22, 783), (19, 27, 1500), (20, 21, 376), (20, 22, 400),
        (21, 24, 480), (22, 26, 551), (23, 25, 1213), (24, 25, 819), (24, 26, 474),
        (26, 27, 1209),
    ]


class Topology:
    """
    Network topology with pre-computed K-shortest paths.

    Stores directed link representation, adjacency, and K=5 shortest paths
    per (src, dst) pair computed via Yen's algorithm with Dijkstra.
    """

    HOP_W = 1.0e6

    def __init__(self, name: str, num_nodes: int, edges: List[Tuple[int, int, int]], k: int = 5,
                 order: str = "km"):
        self.name = name
        self.num_nodes = num_nodes
        self.k = k
        # Candidate-path ordering. "km" (default): K shortest paths by length.
        # "hops": K shortest by number of hops with km as the secondary criterion
        # (#hops ordering, Doherty et al. 2025). Implemented
        # exactly as a lexicographic cost: every link adds HOP_W (> any path length in km) plus its km.
        if order not in ("km", "hops"):
            raise ValueError(f"unknown path order {order!r}")
        self.order = order
        self._hop_w = self.HOP_W if order == "hops" else 0.0

        # Build directed links from undirected edges
        self.link_map: Dict[int, Tuple[int, int, int]] = {}  # link_id -> (src, dst, km)
        self.adj: Dict[int, List[Tuple[int, int, int]]] = {n: [] for n in range(num_nodes)}
        # node_pair_to_link: (src, dst) -> link_id
        self.node_pair_to_link: Dict[Tuple[int, int], int] = {}

        link_id = 0
        for (u, v, d) in edges:
            self.link_map[link_id] = (u, v, d)
            self.adj[u].append((v, d, link_id))
            self.node_pair_to_link[(u, v)] = link_id
            link_id += 1
            self.link_map[link_id] = (v, u, d)
            self.adj[v].append((u, d, link_id))
            self.node_pair_to_link[(v, u)] = link_id
            link_id += 1

        self.num_links = link_id  # directed count

        lengths = [d for (_, _, d) in edges]
        self.avg_link_length = sum(lengths) / len(lengths)
        self.min_link_length = min(lengths)
        self.max_link_length = max(lengths)

        # Pre-compute K shortest paths
        self.k_paths: Dict[int, Dict[int, List[List[int]]]] = {}
        for s in range(num_nodes):
            self.k_paths[s] = {}
            for d in range(num_nodes):
                if s == d:
                    self.k_paths[s][d] = []
                else:
                    self.k_paths[s][d] = self._yen_k_shortest(s, d, k)

    def _dijkstra(self, src: int, dst: int, excluded_nodes: Set[int],
                  excluded_edges: Set[Tuple[int, int]]) -> Optional[List[int]]:
        """Dijkstra shortest path returning list of nodes, or None if unreachable."""
        dist = {src: 0.0}
        prev = {src: None}
        visited = set()
        heap = [(0.0, src)]

        while heap:
            d_u, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == dst:
                # Reconstruct path as node list
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                return path
            for (v, weight, _lid) in self.adj[u]:
                if v in excluded_nodes:
                    continue
                if (u, v) in excluded_edges:
                    continue
                nd = d_u + weight + self._hop_w
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        return None

    def _node_path_to_link_path(self, node_path: List[int]) -> List[int]:
        """Convert a node-sequence path to a link-index path."""
        links = []
        for i in range(len(node_path) - 1):
            u, v = node_path[i], node_path[i + 1]
            links.append(self.node_pair_to_link[(u, v)])
        return links

    def _path_cost(self, node_path: List[int]) -> float:
        """Ranking cost of a node-sequence path: total km (plus HOP_W per link for "hops" order)."""
        cost = 0.0
        for i in range(len(node_path) - 1):
            u, v = node_path[i], node_path[i + 1]
            lid = self.node_pair_to_link[(u, v)]
            cost += self.link_map[lid][2] + self._hop_w
        return cost

    def _yen_k_shortest(self, src: int, dst: int, K: int) -> List[List[int]]:
        """Yen's K-shortest paths algorithm. Returns list of link-index paths."""
        # Find the first shortest path
        first_node_path = self._dijkstra(src, dst, set(), set())
        if first_node_path is None:
            return []

        A = [first_node_path]  # confirmed shortest paths (node lists)
        B = []  # candidate paths: (cost, path)

        for k_i in range(1, K):
            prev_path = A[k_i - 1]
            for i in range(len(prev_path) - 1):
                spur_node = prev_path[i]
                root_path = prev_path[:i + 1]
                root_cost = self._path_cost(root_path) if len(root_path) > 1 else 0.0

                excluded_edges = set()
                for p in A:
                    if len(p) > i and p[:i + 1] == root_path:
                        excluded_edges.add((p[i], p[i + 1]))

                excluded_nodes = set(root_path[:-1])  # exclude root nodes except spur

                spur_path = self._dijkstra(spur_node, dst, excluded_nodes, excluded_edges)
                if spur_path is not None:
                    total_path = root_path[:-1] + spur_path
                    total_cost = self._path_cost(total_path)
                    # Check for duplicate
                    is_dup = False
                    for (bc, bp) in B:
                        if bp == total_path:
                            is_dup = True
                            break
                    if not is_dup:
                        for ap in A:
                            if ap == total_path:
                                is_dup = True
                                break
                    if not is_dup:
                        heapq.heappush(B, (total_cost, total_path))

            if not B:
                break
            _, best_path = heapq.heappop(B)
            A.append(best_path)

        # Convert all to link-index paths
        return [self._node_path_to_link_path(p) for p in A]

    def get_path_length(self, path: List[int]) -> float:
        """Total km for a path given as link indices."""
        return sum(self.link_map[lid][2] for lid in path)

    def get_path_links(self, path: List[int]) -> List[int]:
        """Return the link indices (identity, included for API consistency)."""
        return list(path)

    def get_path_nodes(self, path: List[int]) -> List[int]:
        """Return the node sequence for a path given as link indices."""
        if not path:
            return []
        nodes = [self.link_map[path[0]][0]]
        for lid in path:
            nodes.append(self.link_map[lid][1])
        return nodes

    def get_khop_links(self, node: int, k: int = 2) -> Set[int]:
        """Set of directed link indices within k hops of node."""
        visited_nodes = {node}
        frontier = {node}
        links = set()
        for _ in range(k):
            next_frontier = set()
            for n in frontier:
                for (v, _d, lid) in self.adj[n]:
                    links.add(lid)
                    if v not in visited_nodes:
                        visited_nodes.add(v)
                        next_frontier.add(v)
            frontier = next_frontier
        return links

    def get_neighbors(self, node: int) -> List[int]:
        """List of adjacent node IDs."""
        return [v for (v, _d, _lid) in self.adj[node]]


def create_topology(name: str, k: int = 5, order: str = "km") -> Topology:
    """Factory function to create a Topology by name (k = number of candidate paths)."""
    name_upper = name.upper()
    if name_upper == "NSFNET":
        return Topology("NSFNET", 14, _nsfnet_edges(), k=k, order=order)
    elif name_upper == "USNET":
        return Topology("USNET", 24, _usnet_edges(), k=k, order=order)
    elif name_upper == "EUROPEAN":
        edges = _european_edges()
        # Determine actual node count from edges
        nodes_seen = set()
        for u, v, _ in edges:
            nodes_seen.add(u)
            nodes_seen.add(v)
        num_nodes = max(nodes_seen) + 1
        return Topology("EUROPEAN", num_nodes, edges, k=k, order=order)
    else:
        raise ValueError(f"Unknown topology: {name}")


TOPOLOGY_NAMES = ("NSFNET", "USNET", "EUROPEAN")
