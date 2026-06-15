package org.uma.ed.datastructures.graph;

import org.uma.ed.datastructures.dictionary.Dictionary;
import org.uma.ed.datastructures.dictionary.JDKHashDictionary;
import org.uma.ed.datastructures.list.JDKArrayList;
import org.uma.ed.datastructures.list.List;
import org.uma.ed.datastructures.set.JDKHashSet;
import org.uma.ed.datastructures.set.Set;
import org.uma.ed.datastructures.stack.JDKStack;
import org.uma.ed.datastructures.stack.Stack;

/**
 * This class implements a depth-first traversal on a graph, computing also the corresponding spanning tree.
 *
 * @param <V> The type of the vertices in the graph.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class DFTPaths<V> {
  private final V source;                  // source of exploration
  private final Set<V> visited;            // already visited nodes
  private final List<V> traversal;         // traversal
  private final Dictionary<V, V> parentOf; // spanning tree as parent of dictionary

  public DFTPaths(Traversable<V> graph, V source) {
    this.source = source; // save source

    visited = JDKHashSet.empty();
    traversal = JDKArrayList.empty();
    parentOf = JDKHashDictionary.empty();

    // stack of pending nodes as edges
    Stack<DiEdge<V>> stack = JDKStack.of(DiEdge.of(source, source));

    while (!stack.isEmpty()) {
      DiEdge<V> diEdge = stack.top();
      stack.pop();

      V predecessor = diEdge.source();
      V vertex = diEdge.destination();

      if (!visited.contains(vertex)) {
        visited.insert(vertex);
        traversal.append(vertex);
        parentOf.insert(vertex, predecessor);
        for(V successor : graph.successors(vertex)) {
          if (!visited.contains(successor)) {
            stack.push(DiEdge.of(vertex, successor));
          }
        }
      }
    }
  }

  public static <V> DFTPaths<V> of(Traversable<V> graph, V source) {
    return new DFTPaths<>(graph, source);
  }

  public List<V> traversal() {
    return traversal;
  }

  public List<V> pathTo(V destination) {
    if (!visited.contains(destination)) {
      throw new GraphException("Vertex " + destination + "was not visited");
    }

    V vertex = destination;
    List<V> path = JDKArrayList.of(vertex);
    while (!vertex.equals(source)) {
      vertex = parentOf.valueOf(vertex);
      path.prepend(vertex);
    }

    return path;
  }
}


class DFTPathsTest {
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

    DFTPaths<Character> dftPaths = DFTPaths.of(g, 'A');
    System.out.println("Depth first traversal was: " + dftPaths.traversal());
    System.out.println("Path to B is: " + dftPaths.pathTo('B'));
    System.out.println("Path to D is: " + dftPaths.pathTo('D'));
  }
}