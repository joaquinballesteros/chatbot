package org.uma.ed.datastructures.set;

import java.util.Comparator;
import java.util.Iterator;
import java.util.NoSuchElementException;
import org.uma.ed.datastructures.searchtree.RBT;
import org.uma.ed.datastructures.searchtree.SearchTree;

/**
 * Sets implemented using Red Black Binary Search Trees. Order of elements is defined by provided comparator or natural 
 * order if none is provided.
 *
 * @param <T> Type of elements in set.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class RBTSet<T> extends AbstractSortedSet<T> implements SortedSet<T> {
  private final SearchTree<T> rbTree;

  private RBTSet(RBT<T> rbTree) {
    this.rbTree = rbTree;
  }

  /**
   * Constructs an empty sorted set with order provided by parameter.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of elements in this sorted set.
   */
  public RBTSet(Comparator<T> comparator) {
    this(RBT.empty(comparator));
  }

  /**
   * Constructs an empty sorted set with natural order of elements.
   * <p> Time complexity: O(1)
   */
  public static <T extends Comparable<? super T>> RBTSet<T> empty() {
    return new RBTSet<T>(Comparator.naturalOrder());
  }

  /**
   * Constructs an empty sorted set with order provided by parameter.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of elements in this sorted set.
   */
  public static <T> RBTSet<T> empty(Comparator<T> comparator) {
    return new RBTSet<>(comparator);
  }

  /**
   * Creates a new RBTSet with provided comparator and elements.
   * <p> Time complexity: O(n log n)
   *
   * @param comparator Comparator defining order of elements in new sorted set.
   * @param elements Elements to include in new set.
   * @param <T> Type of elements in new set.
   *
   * @return New RBTSet with provided comparator and elements.
   */
  @SafeVarargs
  public static <T> RBTSet<T> of(Comparator<T> comparator, T... elements) {
    RBTSet<T> rbtSet = new RBTSet<>(comparator);
    rbtSet.insert(elements);
    return rbtSet;
  }

  /**
   * Creates a new RBTSet with natural order and provided elements.
   * <p> Time complexity: O(n log n)
   *
   * @param elements Elements to include in new set.
   * @param <T> Type of elements in new set.
   *
   * @return a new RBTSet with natural order and provided elements
   */
  @SafeVarargs
  public static <T extends Comparable<? super T>> RBTSet<T> of(T... elements) {
    return RBTSet.of(Comparator.naturalOrder(), elements);
  }

  /**
   * Creates a new RBTSet with provided comparator and elements in iterable.
   * <p> Time complexity: O(n log n)
   *
   * @param comparator Comparator defining order of elements in new sorted set.
   * @param iterable iterable with elements to include in new set.
   * @param <T> Type of elements in new set.
   *
   * @return New RBTSet with provided comparator and elements.
   */
  public static <T> RBTSet<T> from(Comparator<T> comparator, Iterable<T> iterable) {
    RBTSet<T> srbtSet = new RBTSet<>(comparator);
    for (T element : iterable) {
      srbtSet.insert(element);
    }
    return srbtSet;
  }

  /**
   * Creates a new RBTSet with natural order and elements in iterable.
   * <p> Time complexity: O(n log n)
   *
   * @param iterable iterable with elements to include in new set.
   * @param <T> Type of elements in new set.
   *
   * @return New RBTSet with provided comparator and elements.
   */
  public static <T extends Comparable<? super T>> RBTSet<T> from(Iterable<T> iterable) {
    return from(Comparator.naturalOrder(), iterable);
  }

  /**
   * Returns a new RBTSet with same elements as argument.
   * <p> Time complexity: O(n)
   *
   * @param that RBTSet to be copied.
   *
   * @return a new RBTSet with same elements as {@code that}.
   */
  public static <T> RBTSet<T> copyOf(RBTSet<T> that) {
    return new RBTSet<>(RBT.copyOf(that.rbTree));
  }

  /**
   * Returns a new RBTSet with same elements as argument.
   * <p> Time complexity: O(n log n)
   *
   * @param that Sorted set to be copied.
   *
   * @return a new RBTSet with same elements as {@code that}.
   */
  public static <T> RBTSet<T> copyOf(SortedSet<T> that) {
    if (that instanceof RBTSet<T> rbtSet) {
      // use specialized version for RBTSet
      return copyOf(rbtSet);
    }
    // todo could be improved as elements in that are already sorted
    RBTSet<T> copy = new RBTSet<>(that.comparator());
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
    return rbTree.comparator();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public boolean isEmpty() {
    return rbTree.isEmpty();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public int size() {
    return rbTree.size();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void insert(T element) {
    rbTree.insert(element);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public boolean contains(T element) {
    return rbTree.contains(element);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void delete(T element) {
    rbTree.delete(element);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public void clear() {
    rbTree.clear();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public T minimum() {
    if (isEmpty()) {
      throw new NoSuchElementException("minimum on empty set");
    }
    return rbTree.minimum();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public T maximum() {
    if (isEmpty()) {
      throw new NoSuchElementException("maximum on empty set");
    }
    return rbTree.maximum();
  }

  /**
   * Iterator over elements in set. Notice that {@code remove} method is not supported. Note also that set should not be
   * modified during iteration as iterator state may become inconsistent.
   *
   * @see Iterable#iterator()
   */
  @Override
  public Iterator<T> iterator() {
    return rbTree.inOrder().iterator();
  }
}
