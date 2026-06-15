package org.uma.ed.datastructures.graph;

import org.uma.ed.datastructures.list.JDKArrayList;
import org.uma.ed.datastructures.list.List;
import org.uma.ed.datastructures.set.JDKHashSet;
import org.uma.ed.datastructures.set.Set;
import org.uma.ed.datastructures.stack.JDKStack;
import org.uma.ed.datastructures.stack.Stack;

/**
 * This class implements a depth-first traversal on a graph.
 *
 * @param <V> The type of the vertices in the graph.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class DFT<V> {
  private final List<V> traversal; // traversal

  public DFT(Traversable<V> graph, V source) {
    // already visited nodes
    Set<V> visited = JDKHashSet.empty();
    traversal = JDKArrayList.empty();

    // stack of pending nodes
    Stack<V> stack = JDKStack.of(source);

    while (!stack.isEmpty()) {
      V vertex = stack.top();
      stack.pop();

      if (!visited.contains(vertex)) {
        visited.insert(vertex);
        traversal.append(vertex);
        for(V successor : graph.successors(vertex)) {
          if (!visited.contains(successor)) {
            stack.push(successor);
          }
        }
      }
    }
  }

  public static <V> DFT<V> of(Traversable<V> graph, V source) {
    return new DFT<>(graph, source);
  }

  public List<V> traversal() {
    return traversal;
  }
}


class DFTTest {
  public static void main(String[] args) {
    Graph<Character> g = DictionaryGraph.empty();
    g.addVertex('A');
    g.addVertex('B');
    g.addVertex('C');
    g.addVertex('D');
    g.addVertex('E');
    g.addVertex('F');
    g.addVertex('G');
    g.addVertex('H');

    g.addEdge('A', 'C');
    g.addEdge('C', 'G');
    g.addEdge('G', 'E');
    g.addEdge('E', 'D');
    g.addEdge('E', 'F');
    g.addEdge('E', 'H');
    g.addEdge('H', 'A');
    g.addEdge('H', 'B');
    g.addEdge('F', 'D');
    g.addEdge('D', 'A');

    DFT<Character> dft = DFT.of(g, 'A');
    System.out.println("Depth first traversal was: " + dft.traversal());
  }
}