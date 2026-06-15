package org.uma.ed.datastructures.graph ;

import org.uma.ed.datastructures.list.ArrayList;
import org.uma.ed.datastructures.list.List;

/**
 * This class is used to compute a topological sorting for directed graphs. Topological sorting for a directed graph is
 * a linear ordering of its vertices such that for every directed edge (u, v), vertex u comes before v in the ordering.
 * Topological Sorting for a graph is not possible if the graph is not a Directed Acyclic Graph (DAG).
 *
 * @param <V> The type of the vertices in the graph.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class TopologicalSorting<V> {

  // EXERCISE: solve this same problem using the reversed graph

  private final List<V> order; // List to store the topological order
  private boolean cycle; // Flag to indicate if the graph contains a cycle

  /**
   * Constructs a new TopologicalSorting object and performs a topological sort on the given directed graph.
   *
   * @param diGraph The directed graph to be topologically sorted.
   */
  public TopologicalSorting(DiGraph<V> diGraph) {
    order = ArrayList.empty();
    cycle = false;

    DiGraph<V> dg = DictionaryDiGraph.copyOf(diGraph); // copy of graph (to be able to delete vertices and edges

    while (!cycle && !dg.isEmpty()) {
      V source = null;
      for (V vertex : dg.vertices()) {
        if (dg.inDegree(vertex) == 0) {
          source = vertex;
          break;
        }
      }

      if (source != null) {
        order.append(source);
        dg.deleteVertex(source); // also deletes corresponding edges
      } else {
        cycle = true;
      }
    }
  }

  /**
   * Checks if the graph contains a cycle.
   *
   * @return true if the graph contains a cycle, false otherwise.
   */
  public boolean hasCycle() {
    return cycle;
  }

  /**
   * Returns the topological order of the vertices in the graph.
   *
   * @return A list of vertices in topological order if the graph does not contain a cycle, null otherwise.
   */
  public List<V> order() {
    return cycle ? null : order;
  }
}
