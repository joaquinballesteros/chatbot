
package org.uma.ed.datastructures.graph;

import java.util.StringJoiner;
import org.uma.ed.datastructures.dictionary.Dictionary;
import org.uma.ed.datastructures.dictionary.HashDictionary;
import org.uma.ed.datastructures.set.HashSet;
import org.uma.ed.datastructures.set.Set;

/**
 * Class for directed graphs implemented with a dictionary.
 *
 * @param <V> Type for vertices in graph
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class DictionaryDiGraph<V> implements DiGraph<V> {
  // dictionary with keys as vertices in DiGraph and values as successor vertices
  private final Dictionary<V, Set<V>> successorsOf;

  public DictionaryDiGraph() {
    successorsOf = HashDictionary.empty();
  }

  /**
   * Creates an empty directed graph.
   *
   * @param <V> Type for vertices in graph.
   *
   * @return An empty directed graph.
   */
  public static <V> DictionaryDiGraph<V> empty() {
    return new DictionaryDiGraph<>();
  }

  /**
   * Creates a directed graph with given vertices and edges.
   *
   * @param vertices vertices to add to graph.
   * @param edges edges to add to graph.
   * @param <V> Type for vertices in graph.
   *
   * @return A DictionaryDiGraph with given vertices and edges.
   */
  public static <V> DictionaryDiGraph<V> of(Set<V> vertices, Set<DiEdge<V>> edges) {
    DictionaryDiGraph<V> diGraph = new DictionaryDiGraph<>();
    for (V vertex : vertices) {
      diGraph.addVertex(vertex);
    }
    for (DiEdge<V> edge : edges) {
      diGraph.addDiEdge(edge.source(), edge.destination());
    }
    return diGraph;
  }

  /**
   * Creates a directed graph with same vertices and edges as given graph.
   *
   * @param diGraph Graph to copy.
   * @param <V> Type for vertices in graph.
   *
   * @return A DictionaryDiGraph with same vertices and edges as given graph.
   */
  public static <V> DictionaryDiGraph<V> copyOf(DiGraph<V> diGraph) {
    return DictionaryDiGraph.of(diGraph.vertices(), diGraph.edges());
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public boolean isEmpty() {
    return successorsOf.isEmpty();
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public void addVertex(V vertex) {
    if (!successorsOf.isDefinedAt(vertex)) {
      successorsOf.insert(vertex, HashSet.empty());
    }
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public void addDiEdge(V source, V destination) {
    Set<V> destinations = successorsOf.valueOf(source);
    if (destinations == null) {
      throw new GraphException("vertex " + source + " is not in graph");
    }
    if (!successorsOf.isDefinedAt(destination)) {
      throw new GraphException("vertex " + destination + " is not in graph");
    }
    destinations.insert(destination);
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public void deleteDiEdge(V source, V destination) {
    Set<V> destinations = successorsOf.valueOf(source);
    if (destinations != null) {
      destinations.delete(destination);
    }
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public void deleteVertex(V vertex) {
    successorsOf.delete(vertex); // remove vertex from graph and all its successors
    // remove all edges where vertex is destination
    for (Set<V> destinations : successorsOf.values()) {
      destinations.delete(vertex);
    }
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public Set<V> vertices() {
    return HashSet.from(successorsOf.keys());
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public Set<DiEdge<V>> edges() {
    Set<DiEdge<V>> edges = HashSet.empty();
    for (Dictionary.Entry<V, Set<V>> entry : successorsOf.entries()) {
      V source = entry.key();
      Set<V> destinations = entry.value();
      for (V destination : destinations) {
        edges.insert(DiEdge.of(source, destination));
      }
    }
    return edges;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public int numberOfVertices() {
    return successorsOf.size();
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public int numberOfEdges() {
    int numberOfEdges = 0;
    for (Set<V> destinations : successorsOf.values()) {
      numberOfEdges += destinations.size();
    }
    return numberOfEdges;
  }

  /**
   * Returns the successors of a vertex in graph (i.e. vertices to which there is an edge from given vertex).
   *
   * @param source vertex for which we want to obtain its successors.
   *
   * @return Successors of a vertex.
   */
  @Override
  public Set<V> successors(V source) {
    Set<V> successors = successorsOf.valueOf(source);
    if (successors == null) {
      throw new GraphException("vertex " + source + " is not in graph");
    }
    return successors;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public Set<V> predecessors(V destination) {
    if (!successorsOf.isDefinedAt(destination)) {
      throw new GraphException("vertex " + destination + " is not in graph");
    }
    Set<V> predecessors = HashSet.empty();
    for (Dictionary.Entry<V, Set<V>> entry : successorsOf.entries()) {
      V source = entry.key();
      Set<V> destinations = entry.value();
      if (destinations.contains(destination)) {
        predecessors.insert(source);
      }
    }
    return predecessors;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public int inDegree(V vertex) {
    int inDegree = 0;
    for (Set<V> destinations : successorsOf.values()) {
      if (destinations.contains(vertex)) {
        inDegree++;
      }
    }
    return inDegree;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public int outDegree(V vertex) {
    Set<V> destinations = successorsOf.valueOf(vertex);
    return destinations == null ? 0 : destinations.size();
  }

  @Override
  public String toString() {
    String className = getClass().getSimpleName();

    StringJoiner verticesSJ = new StringJoiner(", ", "vertices(", ")");
    for (V vertex : vertices()) {
      verticesSJ.add(vertex.toString());
    }

    StringJoiner edgesSJ = new StringJoiner(", ", "edges(", ")");
    for (DiEdge<V> edge : edges()) {
      edgesSJ.add(edge.toString());
    }

    StringJoiner sj = new StringJoiner(", ", className + "(", ")");
    sj.add(verticesSJ.toString());
    sj.add(edgesSJ.toString());
    return sj.toString();
  }
}
