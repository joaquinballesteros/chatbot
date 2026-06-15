package org.uma.ed.datastructures.graph;


import org.uma.ed.datastructures.dictionary.Dictionary;
import org.uma.ed.datastructures.dictionary.JDKHashDictionary;
import org.uma.ed.datastructures.stack.JDKStack;
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
public class BipartiteExam<V> {
  public enum Color { Red, Blue;
    Color opposite() {
      return (this == Red) ? Blue : Red;
    }
  }

  private record Pair<V>(V vertex, Color color) {
    static <V> Pair<V> of(V vertex, Color color) {
      return new Pair<>(vertex, color);
    }
  }

  private boolean bipartite;
  private final Dictionary<V, Color> assignedColor;

  static <V> BipartiteExam<V> of(Graph<V> graph) {
    return new BipartiteExam<>(graph);
  }

  public BipartiteExam(Graph<V> graph) {
    bipartite = true;
    assignedColor = JDKHashDictionary.empty();

    if (!graph.isEmpty()) {
      V source = graph.vertices().iterator().next();

      Stack<Pair<V>> stack = JDKStack.of(Pair.of(source, Color.Red));

      while (bipartite && !stack.isEmpty()) {
        Pair<V> pair = stack.top();
        stack.pop();

        Color color = assignedColor.valueOf(pair.vertex);
        if (color == null) {
          // vertex not in dictionary
          assignedColor.insert(pair.vertex, pair.color);
          Color oppositeColor = pair.color.opposite();
          for (V successor : graph.successors(pair.vertex)) {
            stack.push(Pair.of(successor, oppositeColor));
          }
        } else if (color != pair.color) {
          bipartite = false;
        }
      }
    }
  }

  public boolean isBipartite() {
    return bipartite;
  }

  public Dictionary<V, Color> assignedColor() {
    return bipartite ? assignedColor : null;
  }
}

class Test {
  public static void main(String[] args) {
    Graph<Integer> graph = DictionaryGraph.empty();
    graph.addVertex(1);
    graph.addVertex(2);
    graph.addVertex(3);
    graph.addVertex(4);
    graph.addVertex(5);

    graph.addEdge(1, 2);
    graph.addEdge(1, 3);
    graph.addEdge(2, 4);
    graph.addEdge(2, 5);

    BipartiteExam<Integer> bipartiteExam = BipartiteExam.of(graph);
    if (bipartiteExam.isBipartite()) {
      System.out.println("Graph is bipartite");
      System.out.println("Colors assigned to vertices: " + bipartiteExam.assignedColor());
    } else {
      System.out.println("Graph is not bipartite");
    }
  }
}