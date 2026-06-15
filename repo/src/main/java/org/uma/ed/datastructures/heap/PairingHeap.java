package org.uma.ed.datastructures.heap;

import java.util.Comparator;
import java.util.Iterator;
import java.util.StringJoiner;
import org.uma.ed.datastructures.list.ArrayList;
import org.uma.ed.datastructures.list.LinkedList;
import org.uma.ed.datastructures.list.List;

/**
 * A pairing Heap.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class PairingHeap<T> implements Heap<T> {
  private static final class Node<E> {
    E element;
    List<Node<E>> children;

    // singleton heap
    Node(E element) {
      this.element = element;
      this.children = LinkedList.empty();
    }
  }

  private final Comparator<T> comparator;
  private Node<T> root;
  private int size;

  private PairingHeap(Comparator<T> comparator, Node<T> root, int size) {
    this.comparator = comparator;
    this.root = root;
    this.size = size;
  }

  public PairingHeap(Comparator<T> comparator) {
    this(comparator, null, 0);
  }

  public static <T> PairingHeap<T> empty(Comparator<T> comparator) {
    return new PairingHeap<>(comparator);
  }

  public static <T extends Comparable<? super T>> PairingHeap<T> empty() {
    return new PairingHeap<T>(Comparator.naturalOrder());
  }

  /**
   * Constructs a pairing heap from a list of singleton nodes in O(n) time.
   * @param comparator comparator to use
   * @param nodes list of singleton nodes
   * @param <T> type of elements
   *
   * @return skew heap with elements in nodes
   */
  private static <T> PairingHeap<T> merge(Comparator<T> comparator, ArrayList<Node<T>> nodes) {
    PairingHeap<T> heap = empty(comparator);
    int size = nodes.size();
    if (size > 0) {
      heap.size = size;
      while (size > 1) {
        // merge heaps pairwise
        for (int i = 0, j = 0; i < size - 1; i += 2, j++) {
          Node<T> node1 = nodes.get(i);
          Node<T> node2 = nodes.get(i + 1);
          nodes.set(j, heap.merge(node1, node2));
        }
        // recompute new size
        if (size % 2 == 1) {
          nodes.set(size / 2, nodes.get(size - 1));
          size = size / 2 + 1;
        } else {
          size = size / 2;
        }
      }
      heap.root = nodes.get(0);
    }
    return heap;
  }

  @SafeVarargs
  public static <T> PairingHeap<T> of(Comparator<T> comparator, T... elements) {
    ArrayList<Node<T>> nodes = ArrayList.withCapacity(elements.length);
    for (T element : elements) {
      nodes.append(new Node<>(element));
    }
    return merge(comparator, nodes);
  }

  @SafeVarargs
  public static <T extends Comparable<? super T>> PairingHeap<T> of(T... elements) {
    return of(Comparator.naturalOrder(), elements);
  }

  public static <T> PairingHeap<T> from(Comparator<T> comparator, Iterable<T> iterable) {
    ArrayList<Node<T>> nodes = ArrayList.empty();
    for (T element : iterable) {
      nodes.append(new Node<>(element));
    }
    return merge(comparator, nodes);
  }

  public static <T extends Comparable<? super T>> PairingHeap<T> from(Iterable<T> iterable) {
    return from(Comparator.naturalOrder(), iterable);
  }

  public static <T> PairingHeap<T> copyOf(PairingHeap<T> that) {
    return new PairingHeap<>(that.comparator, copyOf(that.root), that.size);
  }

  private static <T> Node<T> copyOf(Node<T> node) {
    if (node == null) {
      return null;
    } else {
      Node<T> copy = new Node<>(node.element);
      for (Node<T> child : node.children) {
        copy.children.append(copyOf(child));
      }
      return copy;
    }
  }

  @Override
  public Comparator<T> comparator() {
    return comparator;
  }

  @Override
  public boolean isEmpty() {
    return root == null;
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public void clear() {
    root = null;
    size = 0;
  }

  @Override
  public void insert(T element) {
    Node<T> singleton = new Node<>(element);
    root = merge(root, singleton);
    size++;
  }

  private Node<T> merge(Node<T> node1, Node<T> node2) {
    if (node1 == null) {
      return node2;
    }
    if (node2 == null) {
      return node1;
    }

    // force node1 to have smaller root
    if (comparator.compare(node2.element, node1.element) < 0) {
      // swap node1 and node2
      Node<T> temp = node1;
      node1 = node2;
      node2 = temp;
    }

    // Add node2 to list of children in node1
    node1.children.prepend(node2);
    return node1;
  }

  @Override
  public T minimum() {
    if (isEmpty()) {
      throw new EmptyHeapException("minimum on empty heap");
    }
    return root.element;
  }

  // merges heaps pairwise in left to right order, and then merges those in right to
  // left order
  private Node<T> mergeIterator(Iterator<Node<T>> iterator) {
    if (!iterator.hasNext()) {
      return null;
    } else {
      Node<T> node1 = iterator.next();
      if (!iterator.hasNext()) {
        return node1;
      } else {
        Node<T> node2 = iterator.next();
        return merge(merge(node1, node2), mergeIterator(iterator));
      }
    }
  }

  @Override
  public void deleteMinimum() {
    if (isEmpty()) {
      throw new EmptyHeapException("deleteMinimum on empty heap");
    }
    root = mergeIterator(root.children.iterator());
    size--;
  }

  @Override
  public String toString() {
    String className = getClass().getSimpleName();
    StringBuilder sb = new StringBuilder(className).append("(");
    toString(sb, root);
    sb.append(")");
    return sb.toString();
  }

  private static <T> void toString(StringBuilder sb, Node<T> node) {
    if (node != null) {
      String className = node.getClass().getSimpleName();
      sb.append(className).append("(");
      sb.append(node.element);
      StringJoiner sj = new StringJoiner(", ", ", [", "])");
      for (Node<T> child : node.children) {
        StringBuilder childSb = new StringBuilder();
        toString(childSb, child);
        sj.add(childSb.toString());
      }
      sb.append(sj);
    }
  }
}