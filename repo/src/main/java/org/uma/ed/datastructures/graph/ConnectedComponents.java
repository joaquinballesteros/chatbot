package org.uma.ed.datastructures.graph ;

import org.uma.ed.datastructures.dictionary.Dictionary;
import org.uma.ed.datastructures.dictionary.HashDictionary;
import org.uma.ed.datastructures.set.HashSet;
import org.uma.ed.datastructures.set.Set;

/**
 * This class is used to find the connected components of an undirected graph. A connected component of an undirected
 * graph is a subgraph in which any two vertices are connected to each other by paths, and which is connected to no
 * additional vertices in the graph.
 *
 * @param <V> The type of the vertices in the graph.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class ConnectedComponents<V> {
  private final Set<Set<V>> components;
  private final Dictionary<V, Integer> inComponent;

  /**
   * Constructs a new ConnectedComponents object and performs a depth-first traversal on the given graph to find its
   * connected components.
   *
   * @param graph The graph for which the connected components are to be found.
   */
  public ConnectedComponents(Graph<V> graph) {
    components = HashSet.empty();
    inComponent = HashDictionary.empty();

    Set<V> unvisited = HashSet.empty();
    for (V vertex : graph.vertices()) {
      unvisited.insert(vertex);
    }

    for (int numberOfComponent = 0; !unvisited.isEmpty(); numberOfComponent++) {
      V source = unvisited.iterator().next();

      Set<V> component = HashSet.empty();
      for (V vertex : DepthFirstTraversal.of(graph, source).vertices()) {
        component.insert(vertex);
        inComponent.insert(vertex, numberOfComponent);
      }

      components.insert(component);

      for (V vertex : component) {
        unvisited.delete(vertex);
      }
    }
  }

  /**
   * Returns the set of connected components of the graph.
   *
   * @return A set of connected components, each represented as a set of vertices.
   */
  public Set<Set<V>> components() {
    return components;
  }

  /**
   * Checks if two vertices are in the same connected component.
   *
   * @param v The first vertex.
   * @param w The second vertex.
   *
   * @return true if the two vertices are in the same connected component, false otherwise.
   */
  public boolean areConnected(V v, V w) {
    return inComponent.valueOf(v).equals(inComponent.valueOf(w));
  }
}
