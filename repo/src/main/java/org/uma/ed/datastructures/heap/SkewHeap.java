package org.uma.ed.datastructures.heap;

import java.util.Comparator;
import org.uma.ed.datastructures.list.ArrayList;

/**
 * Heap implemented using skew heap-ordered binary trees.
 *
 * @param <T> Type of elements in heap.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class SkewHeap<T> implements Heap<T> {
  private static final class Node<E> {
    E element;
    Node<E> left, right;

    Node(E element, Node<E> left, Node<E> right) {
      this.element = element;
      this.left = left;
      this.right = right;
    }

    Node(E element) {
      this(element, null, null);
    }
  }

  private final Comparator<T> comparator;
  private Node<T> root;
  private int size;

  private SkewHeap(Comparator<T> comparator, Node<T> root, int size) {
    this.comparator = comparator;
    this.root = root;
    this.size = size;
  }

  /**
   * Creates an empty Skew Heap.
   * <p> Time complexity: O(1)
   */
  public SkewHeap(Comparator<T> comparator) {
    this(comparator, null, 0);
  }


  public static <T> SkewHeap<T> empty(Comparator<T> comparator) {
    return new SkewHeap<>(comparator);
  }

  public static <T extends Comparable<? super T>> SkewHeap<T> empty() {
    return new SkewHeap<T>(Comparator.naturalOrder());
  }

  /**
   * Constructs a skew heap from a list of singleton nodes in O(n) time.
   * @param comparator comparator to use
   * @param nodes list of singleton nodes
   * @param <T> type of elements
   *
   * @return skew heap with elements in nodes
   */
  private static <T> SkewHeap<T> merge(Comparator<T> comparator, ArrayList<Node<T>> nodes) {
    SkewHeap<T> heap = empty(comparator);
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
  public static <T> SkewHeap<T> of(Comparator<T> comparator, T... elements) {
    ArrayList<Node<T>> nodes = ArrayList.withCapacity(elements.length);
    for (T element : elements) {
      nodes.append(new Node<>(element));
    }
    return merge(comparator, nodes);
  }

  @SafeVarargs
  public static <T extends Comparable<? super T>> SkewHeap<T> of(T... elements) {
    return of(Comparator.naturalOrder(), elements);
  }

  public static <T> SkewHeap<T> from(Comparator<T> comparator, Iterable<T> iterable) {
    ArrayList<Node<T>> nodes = ArrayList.empty();
    for (T element : iterable) {
      nodes.append(new Node<>(element));
    }
    return merge(comparator, nodes);
  }

  public static <T extends Comparable<? super T>> SkewHeap<T> from(Iterable<T> iterable) {
    return from(Comparator.naturalOrder(), iterable);
  }

  /**
   * <p> Time complexity: O(n)
   */
  public static <T> SkewHeap<T> copyOf(SkewHeap<T> that) {
    return new SkewHeap<>(that.comparator, copyOf(that.root), that.size);
  }

  // copies a tree
  private static <T> Node<T> copyOf(Node<T> node) {
    if (node == null) {
      return null;
    } else {
      return new Node<>(node.element, copyOf(node.left), copyOf(node.right));
    }
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public Comparator<T> comparator() {
    return comparator;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public boolean isEmpty() {
    return root == null;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public int size() {
    return size;
  }

  @Override
  public void clear() {
    root = null;
    size = 0;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n) amortized
   */
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

    Node<T> temp;

    // force node1 to have smaller root
    if (comparator.compare(node2.element, node1.element) < 0) {
      // swap node1 and node2
      temp = node1;
      node1 = node2;
      node2 = temp;
    }

    // merge right spines and swap
    temp = node1.left;
    node1.left = merge(node2, node1.right);
    node1.right = temp;
    return node1;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   *
   * @throws <code>EmptyHeapException</code> if heap stores no element.
   */
  @Override
  public T minimum() {
    if (isEmpty()) {
      throw new EmptyHeapException("minimum on empty heap");
    }
    return root.element;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n) amortized
   *
   * @throws <code>EmptyHeapException</code> if heap stores no element.
   */
  @Override
  public void deleteMinimum() {
    if (isEmpty()) {
      throw new EmptyHeapException("deleteMinimum on empty heap");
    }
    root = merge(root.left, root.right);
    size--;
  }

  /**
   * Returns representation of this heap as a String.
   */
  @Override
  public String toString() {
    String className = getClass().getSimpleName();
    StringBuilder sb = new StringBuilder(className).append("(");
    toString(sb, root);
    sb.append(")");
    return sb.toString();
  }

  private static void toString(StringBuilder sb, Node<?> node) {
    if (node == null) {
      sb.append("null");
    } else {
      String className = node.getClass().getSimpleName();
      sb.append(className).append("(");
      toString(sb, node.left);
      sb.append(", ");
      sb.append(node.element);
      sb.append(", ");
      toString(sb, node.right);
      sb.append(")");
    }
  }
}
