package org.uma.ed.datastructures.set;

import java.util.Comparator;
import java.util.Iterator;
import java.util.NoSuchElementException;
import org.uma.ed.datastructures.searchtree.LLRBT;
import org.uma.ed.datastructures.searchtree.SearchTree;

/**
 * Sets implemented using Red Black Binary Search Trees. Order of elements is defined by provided comparator or natural 
 * order if none is provided.
 *
 * @param <T> Type of elements in set.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class LLRBTSet<T> extends AbstractSortedSet<T> implements SortedSet<T> {
  private final SearchTree<T> llrbTree;

  private LLRBTSet(LLRBT<T> llrbTree) {
    this.llrbTree = llrbTree;
  }

  /**
   * Constructs an empty sorted set with order provided by parameter.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of elements in this sorted set.
   */
  public LLRBTSet(Comparator<T> comparator) {
    this(LLRBT.empty(comparator));
  }

  /**
   * Constructs an empty sorted set with natural order of elements.
   * <p> Time complexity: O(1)
   */
  public static <T extends Comparable<? super T>> LLRBTSet<T> empty() {
    return new LLRBTSet<T>(Comparator.naturalOrder());
  }

  /**
   * Constructs an empty sorted set with order provided by parameter.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of elements in this sorted set.
   */
  public static <T> LLRBTSet<T> empty(Comparator<T> comparator) {
    return new LLRBTSet<>(comparator);
  }

  /**
   * Creates a new LLRBTSet with provided comparator and elements.
   * <p> Time complexity: O(n log n)
   *
   * @param comparator Comparator defining order of elements in new sorted set.
   * @param elements Elements to include in new set.
   * @param <T> Type of elements in new set.
   *
   * @return New LLRBTSet with provided comparator and elements.
   */
  @SafeVarargs
  public static <T> LLRBTSet<T> of(Comparator<T> comparator, T... elements) {
    LLRBTSet<T> llrbtSet = new LLRBTSet<>(comparator);
    llrbtSet.insert(elements);
    return llrbtSet;
  }

  /**
   * Creates a new LLRBTSet with natural order and provided elements.
   * <p> Time complexity: O(n log n)
   *
   * @param elements Elements to include in new set.
   * @param <T> Type of elements in new set.
   *
   * @return a new LLRBTSet with natural order and provided elements
   */
  @SafeVarargs
  public static <T extends Comparable<? super T>> LLRBTSet<T> of(T... elements) {
    return LLRBTSet.of(Comparator.naturalOrder(), elements);
  }

  /**
   * Creates a new LLRBTSet with provided comparator and elements in iterable.
   * <p> Time complexity: O(n log n)
   *
   * @param comparator Comparator defining order of elements in new sorted set.
   * @param iterable iterable with elements to include in new set.
   * @param <T> Type of elements in new set.
   *
   * @return New LLRBTSet with provided comparator and elements.
   */
  public static <T> LLRBTSet<T> from(Comparator<T> comparator, Iterable<T> iterable) {
    LLRBTSet<T> llrbtSet = new LLRBTSet<>(comparator);
    for (T element : iterable) {
      llrbtSet.insert(element);
    }
    return llrbtSet;
  }

  /**
   * Creates a new LLRBTSet with natural order and elements in iterable.
   * <p> Time complexity: O(n log n)
   *
   * @param iterable iterable with elements to include in new set.
   * @param <T> Type of elements in new set.
   *
   * @return New LLRBTSet with provided comparator and elements.
   */
  public static <T extends Comparable<? super T>> LLRBTSet<T> from(Iterable<T> iterable) {
    return from(Comparator.naturalOrder(), iterable);
  }

  /**
   * Returns a new LLRBTSet with same elements as argument.
   * <p> Time complexity: O(n)
   *
   * @param that LLRBTSet to be copied.
   *
   * @return a new LLRBTSet with same elements as {@code that}.
   */
  public static <T> LLRBTSet<T> copyOf(LLRBTSet<T> that) {
    return new LLRBTSet<>(LLRBT.copyOf(that.llrbTree));
  }

  /**
   * Returns a new LLRBTSet with same elements as argument.
   * <p> Time complexity: O(n log n)
   *
   * @param that Sorted set to be copied.
   *
   * @return a new LLRBTSet with same elements as {@code that}.
   */
  public static <T> LLRBTSet<T> copyOf(SortedSet<T> that) {
    if (that instanceof LLRBTSet<T> llrbtSet) {
      // use specialized version for LLRBTSet
      return copyOf(llrbtSet);
    }
    // todo could be improved as elements in that are already sorted
    LLRBTSet<T> copy = new LLRBTSet<>(that.comparator());
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
    return llrbTree.comparator();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public boolean isEmpty() {
    return llrbTree.isEmpty();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public int size() {
    return llrbTree.size();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void insert(T element) {
    llrbTree.insert(element);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public boolean contains(T element) {
    return llrbTree.contains(element);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void delete(T element) {
    llrbTree.delete(element);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public void clear() {
    llrbTree.clear();
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
    return llrbTree.minimum();
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
    return llrbTree.maximum();
  }

  /**
   * Iterator over elements in set. Notice that {@code remove} method is not supported. Note also that set should not be
   * modified during iteration as iterator state may become inconsistent.
   *
   * @see Iterable#iterator()
   */
  @Override
  public Iterator<T> iterator() {
    return llrbTree.inOrder().iterator();
  }
}
