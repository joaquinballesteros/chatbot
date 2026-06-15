package org.uma.ed.datastructures.graph ;

import java.util.Arrays;
import org.uma.ed.datastructures.list.ArrayList;
import org.uma.ed.datastructures.list.LinkedList;
import org.uma.ed.datastructures.list.List;

/**
 * This class is used to compute an Eulerian cycle in a graph using Hierholzer's algorithm.
 * <a href="https://en.wikipedia.org/wiki/Eulerian_path">Wikipedia</a>
 * An Eulerian cycle is a cycle
 * in a graph which visits every edge exactly once. Hierholzer's algorithm is efficient and works by building up a cycle
 * and then repeatedly extending it. The algorithm is only applicable to graphs that have an Eulerian cycle, i.e., all
 * vertices have even degree.
 *
 * @param <V> The type of the vertices in the graph.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class EulerianCycle<V> {
  private final List<V> eulerianCycle; // List to store the Eulerian cycle

  /**
   * Constructs a new EulerianCycle object and computes an Eulerian cycle in the given graph.
   *
   * @param g The graph for which an Eulerian cycle is to be computed.
   */
  public EulerianCycle(Graph<V> g) {
    Graph<V> graph = DictionaryGraph.copyOf(g); // Copy of the graph to allow modifications
    eulerianCycle = eulerianCycle(graph); // Compute the Eulerian cycle
  }

  /**
   * Factory method to create a new EulerianCycle object.
   *
   * @param g The graph for which an Eulerian cycle is to be computed.
   *
   * @return A new EulerianCycle object.
   */
  public static <V> EulerianCycle<V> of(Graph<V> g) {
    return new EulerianCycle<>(g);
  }

  /**
   * Checks if the graph has an Eulerian cycle.
   *
   * @return true if the graph has an Eulerian cycle, false otherwise.
   */
  public boolean isEulerian() {
    return eulerianCycle != null;
  }

  /**
   * Returns the Eulerian cycle of the graph.
   *
   * @return A list of vertices forming the Eulerian cycle if the graph has an Eulerian cycle, null otherwise.
   */
  public List<V> eulerianCycle() {
    return eulerianCycle;
  }

  private static <V> boolean isEulerian(Graph<V> graph) {
    for (V vertex : graph.vertices()) {
      if (graph.degree(vertex) % 2 != 0) {
        return false;
      }
    }
    return true;
  }

  private static <V> void remove(Graph<V> graph, V vertex1, V vertex2) {
    graph.deleteEdge(vertex1, vertex2);
    for (V vertex : Arrays.asList(vertex1, vertex2)) {
      if (graph.degree(vertex) == 0) {
        graph.deleteVertex(vertex);
      }
    }
  }

  private static <V> List<V> extractCycle(Graph<V> graph, V source) {
    List<V> cycle = LinkedList.empty();
    cycle.prepend(source);

    V vertex = source;
    do {
      V successor = graph.successors(vertex).iterator().next();
      cycle.prepend(successor);
      remove(graph, vertex, successor);
      vertex = successor;
    }
    while (!vertex.equals(source));
    return cycle;
  }

  private static <V> void addToCycle(List<V> eulerianCycle, List<V> cycle) {
    V vertex0 = cycle.get(0);

    int index = 0;
    for (V vertex : eulerianCycle) {
      if (vertex.equals(vertex0)) {
        break;
      } else {
        index++;
      }
    }

    if (!eulerianCycle.isEmpty()) {
      eulerianCycle.delete(index);
    }

    for (V vertex : cycle) {
      eulerianCycle.insert(index, vertex);
      index++;
    }
  }

  private static <V> List<V> eulerianCycle(Graph<V> graph) {
    if (!isEulerian(graph)) {
      return null;
    }

    List<V> eulerianCycle = ArrayList.empty();
    V vertex = graph.vertices().iterator().next();

    for (boolean done = false; !done; ) {
      List<V> cycle = extractCycle(graph, vertex);
      addToCycle(eulerianCycle, cycle);
      if (graph.isEmpty()) {
        done = true;
      } else {
        for (V v : eulerianCycle) {
          if (graph.vertices().contains(v)) {
            vertex = v;
            break;
          }
        }
      }
    }
    return eulerianCycle;
  }
}
