package org.uma.ed.datastructures.set;

import java.util.Comparator;
import java.util.Iterator;
import java.util.NoSuchElementException;
import org.uma.ed.datastructures.searchtree.AAT;
import org.uma.ed.datastructures.searchtree.SearchTree;

/**
 * Sets implemented using A. Andersson Search Trees. Order of elements is defined by provided comparator or natural order if none
 * is provided.
 *
 * @param <T> Type of elements in set.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class AATSet<T> extends AbstractSortedSet<T> implements SortedSet<T> {
  private final SearchTree<T> aaTree;

  private AATSet(AAT<T> aaTree) {
    this.aaTree = aaTree;
  }

  /**
   * Constructs an empty sorted set with order provided by parameter.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of elements in this sorted set.
   */
  public AATSet(Comparator<T> comparator) {
    this(AAT.empty(comparator));
  }

  /**
   * Constructs an empty sorted set with natural order of elements.
   * <p> Time complexity: O(1)
   */
  public static <T extends Comparable<? super T>> AATSet<T> empty() {
    return new AATSet<T>(Comparator.naturalOrder());
  }

  /**
   * Constructs an empty sorted set with order provided by parameter.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of elements in this sorted set.
   */
  public static <T> AATSet<T> empty(Comparator<T> comparator) {
    return new AATSet<>(comparator);
  }

  /**
   * Creates a new AATSet with provided comparator and elements.
   * <p> Time complexity: O(n log n)
   *
   * @param comparator Comparator defining order of elements in new sorted set.
   * @param elements Elements to include in new set.
   * @param <T> Type of elements in new set.
   *
   * @return New AATSet with provided comparator and elements.
   */
  @SafeVarargs
  public static <T> AATSet<T> of(Comparator<T> comparator, T... elements) {
    AATSet<T> rbtSet = new AATSet<>(comparator);
    rbtSet.insert(elements);
    return rbtSet;
  }

  /**
   * Creates a new AATSet with natural order and provided elements.
   * <p> Time complexity: O(n log n)
   *
   * @param elements Elements to include in new set.
   * @param <T> Type of elements in new set.
   *
   * @return a new AATSet with natural order and provided elements
   */
  @SafeVarargs
  public static <T extends Comparable<? super T>> AATSet<T> of(T... elements) {
    return AATSet.of(Comparator.naturalOrder(), elements);
  }

  /**
   * Creates a new AATSet with provided comparator and elements in iterable.
   * <p> Time complexity: O(n log n)
   *
   * @param comparator Comparator defining order of elements in new sorted set.
   * @param iterable iterable with elements to include in new set.
   * @param <T> Type of elements in new set.
   *
   * @return New AATSet with provided comparator and elements.
   */
  public static <T> AATSet<T> from(Comparator<T> comparator, Iterable<T> iterable) {
    AATSet<T> srbtSet = new AATSet<>(comparator);
    for (T element : iterable) {
      srbtSet.insert(element);
    }
    return srbtSet;
  }

  /**
   * Creates a new AATSet with natural order and elements in iterable.
   * <p> Time complexity: O(n log n)
   *
   * @param iterable iterable with elements to include in new set.
   * @param <T> Type of elements in new set.
   *
   * @return New AATSet with provided comparator and elements.
   */
  public static <T extends Comparable<? super T>> AATSet<T> from(Iterable<T> iterable) {
    return from(Comparator.naturalOrder(), iterable);
  }

  /**
   * Returns a new AATSet with same elements as argument.
   * <p> Time complexity: O(n)
   *
   * @param that AATSet to be copied.
   *
   * @return a new AATSet with same elements as {@code that}.
   */
  public static <T> AATSet<T> copyOf(AATSet<T> that) {
    return new AATSet<>(AAT.copyOf(that.aaTree));
  }

  /**
   * Returns a new AATSet with same elements as argument.
   * <p> Time complexity: O(n log n)
   *
   * @param that Sorted set to be copied.
   *
   * @return a new AATSet with same elements as {@code that}.
   */
  public static <T> AATSet<T> copyOf(SortedSet<T> that) {
    if (that instanceof AATSet<T> rbtSet) {
      // use specialized version for AATSet
      return copyOf(rbtSet);
    }
    // todo could be improved as elements in that are already sorted
    AATSet<T> copy = new AATSet<>(that.comparator());
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
    return aaTree.comparator();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public boolean isEmpty() {
    return aaTree.isEmpty();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public int size() {
    return aaTree.size();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void insert(T element) {
    aaTree.insert(element);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public boolean contains(T element) {
    return aaTree.contains(element);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void delete(T element) {
    aaTree.delete(element);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public void clear() {
    aaTree.clear();
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
    return aaTree.minimum();
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
    return aaTree.maximum();
  }

  /**
   * Iterator over elements in set. Notice that {@code remove} method is not supported. Note also that set should not be
   * modified during iteration as iterator state may become inconsistent.
   *
   * @see Iterable#iterator()
   */
  @Override
  public Iterator<T> iterator() {
    return aaTree.inOrder().iterator();
  }
}
