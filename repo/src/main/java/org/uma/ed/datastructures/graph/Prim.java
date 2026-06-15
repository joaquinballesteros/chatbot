package org.uma.ed.datastructures.graph ;

import java.util.Iterator;
import org.uma.ed.datastructures.list.LinkedList;
import org.uma.ed.datastructures.list.List;
import org.uma.ed.datastructures.set.HashSet;
import org.uma.ed.datastructures.set.Set;


/**
 * This class implements Prim's algorithm for computing the minimum spanning tree of a weighted graph. A minimum
 * spanning tree of a graph is a subgraph that connects all vertices in the graph, has no cycles, and the total weight
 * of its edges is as small as possible. Prim's algorithm is a greedy algorithm that starts from a single vertex and
 * grows the tree by adding the smallest edge that connects a vertex in the tree to a vertex outside the tree.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class Prim {
  /**
   * Computes the minimum spanning tree of a weighted graph using Prim's algorithm.
   *
   * @param weightedGraph The weighted graph for which the minimum spanning tree is to be computed.
   *
   * @return A list of weighted edges that form the minimum spanning tree of the graph.
   */
  public static <V> List<WeightedEdge<V, Integer>> prim(
      WeightedGraph<V, Integer> weightedGraph) {
    List<WeightedEdge<V, Integer>> tree = LinkedList.empty();
    if (!weightedGraph.isEmpty()) {
      // initialization
      Set<V> T = HashSet.empty();
      Set<V> R = HashSet.empty();

      Iterator<V> it = weightedGraph.vertices().iterator();
      if (it.hasNext()) {
        T.insert(it.next());

        while (it.hasNext()) {
          R.insert(it.next());
        }
      }

      // Initial set with all edges
      Set<WeightedEdge<V, Integer>> edges = HashSet.empty();
      for (WeightedEdge<V, Integer> edge : weightedGraph.edges()) {
        edges.insert(edge);
      }

      while (!R.isEmpty()) {
        WeightedEdge<V, Integer> bestEdge = null;
        V bestInR = null;

        for (WeightedEdge<V, Integer> edge : edges) {
          V inR = connects(T, R, edge);

          if (inR != null && (bestEdge == null || edge.weight().compareTo(bestEdge.weight()) < 0)) {
            bestEdge = edge;
            bestInR = inR;
          }
        }

        tree.append(bestEdge);
        edges.delete(bestEdge);

        T.insert(bestInR);
        R.delete(bestInR);
      }
    }
    return tree;
  }

  /**
   * Checks if an edge connects a vertex in one set to a vertex in another set.
   *
   * @param set1 The first set of vertices.
   * @param set2 The second set of vertices.
   * @param edge The edge to be checked.
   *
   * @return The vertex in the second set that is connected by the edge to a vertex in the first set, or null if no such
   * vertex exists.
   */
  static <V> V connects(Set<V> set1, Set<V> set2, WeightedEdge<V, ?> edge) {
    V vertex1 = edge.vertex1();
    V vertex2 = edge.vertex2();

    V inS2 = null;

    if (set1.contains(vertex1) && set2.contains(vertex2)) {
      inS2 = vertex2;
    } else if (set1.contains(vertex2) && set2.contains(vertex1)) {
      inS2 = vertex1;
    }
    return inS2;
  }
}
