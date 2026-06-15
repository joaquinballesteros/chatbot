package org.uma.ed.datastructures.set;

import java.util.Comparator;
import java.util.Iterator;
import java.util.NoSuchElementException;
import org.uma.ed.datastructures.searchtree.BST;
import org.uma.ed.datastructures.searchtree.SearchTree;

/**
 * Sets implemented using Binary Search Trees. Order of elements is defined by provided comparator or natural order if
 * none is provided.
 *
 * @param <T> Type of elements in set.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class BSTSet<T> extends AbstractSortedSet<T> implements SortedSet<T> {
  private final SearchTree<T> binarySearchTree;

  private BSTSet(BST<T> binarySearchTree) {
    this.binarySearchTree = binarySearchTree;
  }

  /**
   * Constructs an empty sorted set with order provided by parameter.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of elements in this sorted set.
   */
  public BSTSet(Comparator<T> comparator) {
    this(BST.empty(comparator));
  }

  /**
   * Constructs an empty sorted set with natural order of elements.
   * <p> Time complexity: O(1)
   */
  public static <T extends Comparable<? super T>> BSTSet<T> empty() {
    return new BSTSet<T>(Comparator.naturalOrder());
  }

  /**
   * Constructs an empty sorted set with order provided by parameter.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of elements in this sorted set.
   */
  public static <T> BSTSet<T> empty(Comparator<T> comparator) {
    return new BSTSet<>(comparator);
  }

  /**
   * Creates a new BSTSet with provided comparator and elements.
   * <p> Time complexity: O(n²)
   *
   * @param comparator Comparator defining order of elements in new sorted set.
   * @param elements Elements to include in new set.
   * @param <T> Type of elements in new set.
   *
   * @return New BSTSet with provided comparator and elements.
   */
  @SafeVarargs
  public static <T> BSTSet<T> of(Comparator<T> comparator, T... elements) {
    BSTSet<T> bstSet = new BSTSet<>(comparator);
    bstSet.insert(elements);
    return bstSet;
  }

  /**
   * Creates a new BSTSet with natural order and provided elements.
   * <p> Time complexity: O(n²)
   *
   * @param elements Elements to include in new set.
   * @param <T> Type of elements in new set.
   *
   * @return a new BSTSet with natural order and provided elements
   */
  @SafeVarargs
  public static <T extends Comparable<? super T>> BSTSet<T> of(T... elements) {
    return BSTSet.of(Comparator.naturalOrder(), elements);
  }

  /**
   * Creates a new BSTSet with provided comparator and elements in iterable.
   * <p> Time complexity: O(n²)
   *
   * @param comparator Comparator defining order of elements in new sorted set.
   * @param iterable iterable with elements to include in new set.
   * @param <T> Type of elements in new set.
   *
   * @return New BSTSet with provided comparator and elements.
   */
  public static <T> BSTSet<T> from(Comparator<T> comparator, Iterable<T> iterable) {
    BSTSet<T> bstSet = new BSTSet<>(comparator);
    for (T element : iterable) {
      bstSet.insert(element);
    }
    return bstSet;
  }

  /**
   * Creates a new BSTSet with natural order and elements in iterable.
   * <p> Time complexity: O(n²)
   *
   * @param iterable iterable with elements to include in new set.
   * @param <T> Type of elements in new set.
   *
   * @return New BSTSet with provided comparator and elements.
   */
  public static <T extends Comparable<? super T>> BSTSet<T> from(Iterable<T> iterable) {
    return from(Comparator.naturalOrder(), iterable);
  }

  /**
   * Returns a new BSTSet with same elements as argument.
   * <p> Time complexity: O(n)
   *
   * @param that BSTSet to be copied.
   *
   * @return a new BSTSet with same elements as {@code that}.
   */
  public static <T> BSTSet<T> copyOf(BSTSet<T> that) {
    return new BSTSet<>(BST.copyOf(that.binarySearchTree));
  }

  /**
   * Returns a new BSTSet with same elements as argument.
   * <p> Time complexity: O(n²)
   *
   * @param that Sorted set to be copied.
   *
   * @return a new BSTSet with same elements as {@code that}.
   */
  public static <T> BSTSet<T> copyOf(SortedSet<T> that) {
    if (that instanceof BSTSet<T> bstSet) {
      // use specialized version for BSTSet
      return copyOf(bstSet);
    }
    // todo could be improved as elements in that are already sorted
    BSTSet<T> copy = new BSTSet<>(that.comparator());
    for (T element : that) {
      copy.insert(element);
    }
    return copy;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public Comparator<T> comparator() {
    return binarySearchTree.comparator();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public boolean isEmpty() {
    return binarySearchTree.isEmpty();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public int size() {
    return binarySearchTree.size();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: from O(log n) to O(n)
   */
  @Override
  public void insert(T element) {
    binarySearchTree.insert(element);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: from O(log n) to O(n)
   */
  @Override
  public boolean contains(T element) {
    return binarySearchTree.contains(element);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: from O(log n) to O(n)
   */
  @Override
  public void delete(T element) {
    binarySearchTree.delete(element);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public void clear() {
    binarySearchTree.clear();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: from O(log n) to O(n)
   */
  @Override
  public T minimum() {
    if (isEmpty()) {
      throw new NoSuchElementException("minimum on empty set");
    }
    return binarySearchTree.minimum();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: from O(log n) to O(n)
   */
  @Override
  public T maximum() {
    if (isEmpty()) {
      throw new NoSuchElementException("maximum on empty set");
    }
    return binarySearchTree.maximum();
  }

  /**
   * Iterator over elements in set. Notice that {@code remove} method is not supported. Note also that set should not be
   * modified during iteration as iterator state may become inconsistent.
   *
   * @see Iterable#iterator()
   */
  @Override
  public Iterator<T> iterator() {
    return binarySearchTree.inOrder().iterator();
  }
}
