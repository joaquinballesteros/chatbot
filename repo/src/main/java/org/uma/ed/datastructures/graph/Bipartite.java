package org.uma.ed.datastructures.graph ;

import org.uma.ed.datastructures.dictionary.Dictionary;
import org.uma.ed.datastructures.dictionary.HashDictionary;
import org.uma.ed.datastructures.stack.LinkedStack;
import org.uma.ed.datastructures.stack.Stack;

/**
 * This class is used to test if a given graph is bipartite using depth-first traversal. A bipartite graph is a graph
 * whose vertices can be divided into two disjoint sets such that every edge connects a vertex in the first set to one
 * in the second set.
 *
 * @param <V> The type of the vertices in the graph.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class Bipartite<V> {
  private record Pair<V>(V vertex, Boolean color) {
    /**
     * Factory method to create a new Pair instance.
     *
     * @param vertex The vertex of the Pair.
     * @param color The color assigned to the vertex.
     *
     * @return A new Pair instance with the provided vertex and color.
     */
    static <V> Pair<V> of(V vertex, Boolean color) {
      return new Pair<>(vertex, color);
    }
  }

  private final Dictionary<V, Boolean> assignedColor; // Dictionary to store the color assigned to each vertex
  private boolean bipartite = true; // Flag to indicate if the graph is bipartite

  /**
   * Constructs a new Bipartite object and performs a depth-first traversal on the given graph to check if it is
   * bipartite.
   *
   * @param graph The graph to be checked for bipartiteness.
   */
  public Bipartite(Graph<V> graph) {
    Stack<Pair<V>> stack = LinkedStack.empty();
    assignedColor = HashDictionary.empty();

    V source = graph.vertices().iterator().next();
    stack.push(Pair.of(source, true));

    while (!stack.isEmpty()) {
      Pair<V> pair = stack.top();
      stack.pop();

      Boolean color = assignedColor.valueOf(pair.vertex);
      if (color == null) { // pair.vertex was unvisited
        assignedColor.insert(pair.vertex, pair.color);
      } else if (color != pair.color) {
        bipartite = false;
        break;
      }

      boolean inverseColor = !pair.color;
      for (V vertex : graph.successors(pair.vertex)) {
        Boolean vertexColor = assignedColor.valueOf(vertex);
        if (vertexColor == null) { // vertex is unvisited
          stack.push(Pair.of(vertex, inverseColor));
        } else if (vertexColor != inverseColor) {
          bipartite = false;
          break;
        }
      }
    }
  }

  /**
   * Factory method to create a new Bipartite instance and check if the given graph is bipartite.
   *
   * @param graph The graph to be checked for bipartiteness.
   *
   * @return A new Bipartite instance that has performed a depth-first traversal on the given graph to check if it is
   * bipartite.
   */
  public static <V> Bipartite<V> of(Graph<V> graph) {
    return new Bipartite<>(graph);
  }

  /**
   * Checks if the graph is bipartite.
   *
   * @return true if the graph is bipartite, false otherwise.
   */
  public boolean isBipartite() {
    return bipartite;
  }

  /**
   * Returns the colors assigned to the vertices during the depth-first traversal. The colors are represented as Boolean
   * values, with true and false representing the two different sets of the bipartite graph.
   *
   * @return A Dictionary mapping each vertex to its assigned color.
   */
  public Dictionary<V, Boolean> colors() {
    return assignedColor;
  }
}