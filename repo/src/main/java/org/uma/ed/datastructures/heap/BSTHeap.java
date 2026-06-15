package org.uma.ed.datastructures.heap;

import java.util.Comparator;

/**
 * Invariant for the data structure: For each node in the tree, elements in left child are smaller than element at node
 * and elements in right child are larger than OR EQUAL to element at node. As we can see, these binary search trees CAN
 * store repeated elements.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class BSTHeap<T> implements Heap<T> {
  private static final class Node<E> {
    E element;
    Node<E> left, right;

    Node(E element) {
      this.element = element;
      this.left = null;
      this.right = null;
    }
  }

  private final Comparator<T> comparator;
  private Node<T> root; // reference to node at root of BST or null if heap is empty
  private int size;     // number of elements in heap

  private BSTHeap(Comparator<T> comparator, Node<T> root, int size) {
    this.comparator = comparator;
    this.root = root;
    this.size = size;
  }

  // Builds an empty heap (represented as an empty BST)
  public BSTHeap(Comparator<T> comparator) {
    this(comparator, null, 0);
  }

  public static <T> BSTHeap<T> empty(Comparator<T> comparator) {
    return new BSTHeap<>(comparator);
  }

  public static <T extends Comparable<? super T>> BSTHeap<T> empty() {
    return new BSTHeap<T>(Comparator.naturalOrder());
  }

  @SafeVarargs
  public static <T> BSTHeap<T> of(Comparator<T> comparator, T... elements) {
    BSTHeap<T> heap = empty(comparator);
    for (T element : elements) {
      heap.insert(element);
    }
    return heap;
  }

  @SafeVarargs
  public static <T extends Comparable<? super T>> BSTHeap<T> of(T... elements) {
    return of(Comparator.naturalOrder(), elements);
  }

  public static <T> BSTHeap<T> from(Comparator<T> comparator, Iterable<T> iterable) {
    BSTHeap<T> heap = empty(comparator);
    for (T element : iterable) {
      heap.insert(element);
    }
    return heap;
  }

  public static <T extends Comparable<? super T>> BSTHeap<T> from(Iterable<T> iterable) {
    return from(Comparator.naturalOrder(), iterable);
  }

  /**
   * <p> Time complexity: O(n)
   */
  public static <T> BSTHeap<T> copyOf(BSTHeap<T> that) {
    return new BSTHeap<>(that.comparator, copyOf(that.root), that.size);
  }

  // copies a tree
  private static <T> Node<T> copyOf(Node<T> node) {
    if (node == null) {
      return null;
    } else {
      Node<T> copy = new Node<>(node.element);
      copy.left = copyOf(node.left);
      copy.right = copyOf(node.right);
      return copy;
    }
  }

  @Override
  public Comparator<T> comparator() {
    return comparator;
  }

  // Checks if heap is empty
  @Override
  public boolean isEmpty() {
    return root == null;
  }

  // Returns total number of elements stored in heap
  @Override
  public int size() {
    return size;
  }

  @Override
  public void clear() {
    root = null;
    size = 0;
  }

  // Inserts a new element in the heap
  @Override
  public void insert(T element) {
    root = insert(root, element);
    size++;
  }

  private Node<T> insert(Node<T> node, T element) {
    if (node == null) {
      node = new Node<>(element);
    } else if (comparator.compare(element, node.element) < 0) {
      node.left = insert(node.left, element);
    } else {
      node.right = insert(node.right, element);
    }
    return node;
  }

  /**
   * Returns element with maximum priority in heap (i.e., the one with a minimum value). If there are several elements
   * with such minimum value, we return the one that got inserted firstly
   */
  @Override
  public T minimum() {
    if (isEmpty()) {
      throw new EmptyHeapException("minimum on empty heap");
    }
    Node<T> node = root;
    while (node.left != null) {
      node = node.left;
    }
    return node.element;
  }

  /**
   * Deletes element with maximum priority from heap (i.e., the one with a minimum value). If there are several elements
   * with such minimum value, we delete the one that got inserted firstly
   */
  @Override
  public void deleteMinimum() {
    if (isEmpty()) {
      throw new EmptyHeapException("deleteMinimum on empty heap");
    }
    Node<T> parent = null;
    Node<T> node = root;
    while (node.left != null) {
      parent = node;
      node = node.left;
    }
    if (parent == null) {
      root = root.right;
    } else {
      parent.left = node.right;
    }
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
