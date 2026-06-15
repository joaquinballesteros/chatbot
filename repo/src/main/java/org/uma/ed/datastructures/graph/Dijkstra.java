
package org.uma.ed.datastructures.graph ;

import org.uma.ed.datastructures.dictionary.Dictionary;
import org.uma.ed.datastructures.dictionary.HashDictionary;
import org.uma.ed.datastructures.list.LinkedList;
import org.uma.ed.datastructures.list.List;
import org.uma.ed.datastructures.priorityqueue.BinaryHeapPriorityQueue;
import org.uma.ed.datastructures.priorityqueue.PriorityQueue;
import org.uma.ed.datastructures.set.HashSet;
import org.uma.ed.datastructures.set.Set;
import org.uma.ed.datastructures.tuple.Tuple2;

/**
 * Class for computing shortest paths in a weighted graph using Dijkstra's algorithm.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class Dijkstra {
  // Class for representing an extension of a path from vertex source to
  // vertex destination and total cost of reaching destination.
  // This class implements Comparable interface to allow sorting of extensions based on total cost.
  record Extension<V>(V source, V destination, Integer totalCost) implements Comparable<Extension<V>> {
    @Override
    // Best extension is the one with the smallest total cost.
    // Will be used later by the priority queue.
    public int compareTo(Extension that) {
      return this.totalCost.compareTo(that.totalCost);
    }

    static <V> Extension<V> of(V source, V destination, Integer totalCost) {
      return new Extension<>(source, destination, totalCost);
    }
  }

  /**
   * Computes costs of shortest paths from a source vertex to all other vertices in a weighted graph.
   *
   * @param weightedGraph The weighted graph.
   * @param source The source vertex.
   * @param <V> The type of the vertices in the graph.
   *
   * @return a dictionary where keys are vertices and values are the minimum cost to reach them from the source.
   */
  public static <V> Dictionary<V, Integer> dijkstra(
      WeightedGraph<V, Integer> weightedGraph, V source) {
    // Put all vertices in graph (except for source) in set
    // of vertices for which we don't know yet optimal path
    Set<V> vertices = HashSet.empty();
    for (V vertex : weightedGraph.vertices()) {
      if (!vertex.equals(source)) {
        vertices.insert(vertex);
      }
    }

    // Put only source in set of vertices for which we know optimal path
    Set<V> verticesOpt = HashSet.empty();
    verticesOpt.insert(source);

    // Annotate in dictionary optimal cost for reaching source
    Dictionary<V, Integer> costOpt = HashDictionary.empty();
    costOpt.insert(source, 0);

    // while we don't know all optimal paths
    while (!vertices.isEmpty()) {
      // Compute all extensions from a vertex in verticesOpt to
      // another in vertices. Notice that this priority queue is 
      // sorted according to total costs of extensions
      PriorityQueue<Extension<V>> extensions = BinaryHeapPriorityQueue.empty();

      for (V v : verticesOpt) {
        for (WeightedGraph.Successor<V, Integer> successor : weightedGraph.successors(v)) {
          V u = successor.vertex();
          if (vertices.contains(u)) {
            Integer weight = successor.weight();
            // Notice that cost for reaching u is that of reaching v + weight of edge
            extensions.enqueue(Extension.of(v, u, costOpt.valueOf(v) + weight));
          }
        }
      }

      // The first extension in the priority queue is the one with
      // minimal total cost
      Extension<V> bestExtension = extensions.first();
      V vMin = bestExtension.destination();

      // Delete new reached vertex from vertices and add it to verticesOpt
      vertices.delete(vMin);
      verticesOpt.insert(vMin);
      // Annotate its optimal total cost in dictionary
      costOpt.insert(vMin, bestExtension.totalCost());
    }
    return costOpt;
  }

  /**
   * Computes shortest paths (and their costs) from a source vertex to all other vertices in a weighted graph.
   *
   * @param weightedGraph The weighted graph.
   * @param source The source vertex.
   * @param <V> The type of the vertices in the graph.
   *
   * @return a dictionary where keys are vertices and values are pairs with the minimum cost to reach them from the
   * source and the path to reach them.
   */
  public static <V> Dictionary<V, Tuple2<Integer, List<V>>> dijkstraPaths(
      WeightedGraph<V, Integer> weightedGraph, V source) {
    record Extension<V>(V source, V destination, Integer totalCost, List<V> path) implements Comparable<Extension<V>> {
      @Override
      // Best extension is the one with the smallest total cost.
      // Will be used later by the priority queue.
      public int compareTo(Extension that) {
        return this.totalCost.compareTo(that.totalCost);
      }

      static <V> Extension<V> of(V source, V destination, Integer totalCost, List<V> path) {
        return new Extension<>(source, destination, totalCost, path);
      }
    }

    Set<V> vertices = HashSet.empty();
    for (V v : weightedGraph.vertices()) {
      if (!v.equals(source)) {
        vertices.insert(v);
      }
    }

    Set<V> verticesOpt = HashSet.empty();
    verticesOpt.insert(source);

    Dictionary<V, Tuple2<Integer, List<V>>> costOpt = HashDictionary.empty();
    List<V> path = LinkedList.empty();
    path.append(source);
    costOpt.insert(source, Tuple2.of(0, path));

    while (!vertices.isEmpty()) {
      PriorityQueue<Extension<V>> extensions = BinaryHeapPriorityQueue.empty();

      for (V v : verticesOpt) {
        for (WeightedGraph.Successor<V, Integer> successor : weightedGraph.successors(v)) {
          V u = successor.vertex();
          if (vertices.contains(u)) {
            Integer weight = successor.weight();

            Tuple2<Integer, List<V>> costPath = costOpt.valueOf(v);
            Integer costOpt_v = costPath._1();
            List<V> pathTo_v = costPath._2();

            List<V> pathTo_u = LinkedList.empty();
            for (V vertex : pathTo_v) {
              pathTo_u.append(vertex);
            }
            pathTo_u.append(u);

            extensions.enqueue(Extension.of(v, u, costOpt_v + weight, pathTo_u));
          }
        }
      }

      Extension<V> bestExtension = extensions.first();
      V vMin = bestExtension.destination;

      vertices.delete(vMin);
      verticesOpt.insert(vMin);
      costOpt.insert(vMin, Tuple2.of(bestExtension.totalCost, bestExtension.path));
    }
    return costOpt;
  }
}
