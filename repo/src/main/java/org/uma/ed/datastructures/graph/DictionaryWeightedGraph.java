package org.uma.ed.datastructures.graph ;

import java.util.StringJoiner;
import org.uma.ed.datastructures.dictionary.Dictionary;
import org.uma.ed.datastructures.dictionary.HashDictionary;
import org.uma.ed.datastructures.set.HashSet;
import org.uma.ed.datastructures.set.Set;

/**
 * Undirected weighted graph implemented with a dictionary from vertices (sources) to another dictionary from vertices
 * (destinations) to weights
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class DictionaryWeightedGraph<V, W> implements WeightedGraph<V, W> {

  /**
   * Each vertex is associated to a dictionary containing associations from each successor to its weight
   */
  private final Dictionary<V, Dictionary<V, W>> dictionaryOf;

  /**
   * Creates an empty undirected weighted graph.
   */
  public DictionaryWeightedGraph() {
    dictionaryOf = HashDictionary.empty();
  }

  /**
   * Creates an empty undirected weighted graph.
   *
   * @param <V> Type for vertices in weighted graph.
   * @param <W> Type for weights in edges.
   *
   * @return An empty DictionaryWeightedGraph.
   */
  public static <V, W> DictionaryWeightedGraph<V, W> empty() {
    return new DictionaryWeightedGraph<>();
  }

  /**
   * Creates an undirected weighted graph with given vertices and edges.
   *
   * @param vertices vertices to add to weighted graph.
   * @param edges edges to add to weighted graph.
   * @param <V> Type for vertices in weighted graph.
   * @param <W> Type for weights in edges.
   *
   * @return A DictionaryWeightedGraph with given vertices and edges.
   */
  public static <V, W> DictionaryWeightedGraph<V, W> of(Set<V> vertices, Set<WeightedEdge<V, W>> edges) {
    DictionaryWeightedGraph<V, W> weightedGraph = new DictionaryWeightedGraph<>();
    for (V vertex : vertices) {
      weightedGraph.addVertex(vertex);
    }

    for (WeightedEdge<V, W> edge : edges) {
      weightedGraph.addEdge(edge.vertex1(), edge.vertex2(), edge.weight());
    }

    return weightedGraph;
  }

  /**
   * Creates an undirected weighted graph with same vertices and edges as given weighted graph.
   *
   * @param graph Weighted graph to copy vertices and edges from.
   * @param <V> Type for vertices in weighted graph.
   * @param <W> Type for weights in edges.
   *
   * @return A DictionaryWeightedGraph with same vertices and edges as given weighted graph.
   */
  public static <V, W> DictionaryWeightedGraph<V, W> copyOf(
      WeightedGraph<V, W> graph) {
    return DictionaryWeightedGraph.of(graph.vertices(), graph.edges());
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public boolean isEmpty() {
    return dictionaryOf.isEmpty();
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public void addVertex(V vertex) {
    if (!dictionaryOf.isDefinedAt(vertex)) {
      dictionaryOf.insert(vertex, HashDictionary.empty());
    }
  }

  /**
   * {@inheritDoc}
   */
  public void deleteVertex(V vertex) {
    dictionaryOf.delete(vertex);

    for (Dictionary<V, W> weightsDictionary : dictionaryOf.values()) {
      weightsDictionary.delete(vertex);
    }
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public void addEdge(V vertex1, V vertex2, W weight) {
    Dictionary<V, W> weightsDictionary1 = dictionaryOf.valueOf(vertex1);
    if (weightsDictionary1 == null) {
      throw new GraphException("vertex " + vertex1 + " is not in graph");
    }
    Dictionary<V, W> weightsDictionary2 = dictionaryOf.valueOf(vertex2);
    if (weightsDictionary2 == null) {
      throw new GraphException("vertex " + vertex2 + " is not in graph");
    }

    weightsDictionary1.insert(vertex2, weight);
    weightsDictionary2.insert(vertex1, weight);
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public void deleteEdge(V vertex1, V vertex2) {
    Dictionary<V, W> weightsDictionary1 = dictionaryOf.valueOf(vertex1);
    if (weightsDictionary1 != null) {
      weightsDictionary1.delete(vertex2);
    }
    Dictionary<V, W> weightsDictionary2 = dictionaryOf.valueOf(vertex2);
    if (weightsDictionary2 != null) {
      weightsDictionary2.delete(vertex1);
    }
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public Set<Successor<V, W>> successors(V vertex) {
    Dictionary<V, W> weightsDictionary = dictionaryOf.valueOf(vertex);
    if (weightsDictionary == null) {
      throw new GraphException("vertex " + vertex + " is not in graph");
    }

    Set<Successor<V, W>> successors = HashSet.empty();
    for (Dictionary.Entry<V, W> entry : weightsDictionary.entries()) {
      V destination = entry.key();
      W weight = entry.value();
      successors.insert(Successor.of(destination, weight));
    }
    return successors;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public Set<V> vertices() {
    return HashSet.from(dictionaryOf.keys());
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public Set<WeightedEdge<V, W>> edges() {
    Set<WeightedEdge<V, W>> weightedEdges = HashSet.empty();
    for (Dictionary.Entry<V, Dictionary<V, W>> entry1 : dictionaryOf.entries()) {
      V source = entry1.key();
      Dictionary<V, W> weightsDictionary = entry1.value();
      for (Dictionary.Entry<V, W> entry2 : weightsDictionary.entries()) {
        V destination = entry2.key();
        W weight = entry2.value();
        weightedEdges.insert(WeightedEdge.of(source, destination, weight));
      }
    }
    return weightedEdges;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public int numberOfVertices() {
    return dictionaryOf.size();
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public int numberOfEdges() {
    int numberOfEdges = 0;
    for (Dictionary<V, W> weightsDictionary : dictionaryOf.values()) {
      numberOfEdges += weightsDictionary.size();
    }
    return numberOfEdges / 2; // each edge has been counted twice
  }

  @Override
  public String toString() {
    String className = getClass().getSimpleName();

    StringJoiner verticesSJ = new StringJoiner(", ", "vertices(", ")");
    for (V vertex : vertices()) {
      verticesSJ.add(vertex.toString());
    }

    StringJoiner weightedEdgesSJ = new StringJoiner(", ", "edges(", ")");
    for (WeightedEdge<V, W> weightedEdge : edges()) {
      weightedEdgesSJ.add(weightedEdge.toString());
    }

    StringJoiner sj = new StringJoiner(", ", className + "(", ")");
    sj.add(verticesSJ.toString());
    sj.add(weightedEdgesSJ.toString());
    return sj.toString();
  }
}
