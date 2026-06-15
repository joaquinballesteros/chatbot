package org.uma.ed.datastructures.set;

import java.util.Iterator;
import org.uma.ed.datastructures.hashtable.SeparateChainingHashTable;

/**
 * Sets implemented using separate chaining hash tables. Notice that elements should redefine
 * {@link java.lang.Object#equals} and {@link java.lang.Object#hashCode} methods properly.
 *
 * @param <T> Type of elements in set.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class SeparateChainingHashSet<T> extends AbstractSet<T> implements Set<T> {
  private final SeparateChainingHashTable<T> hashTable;

  /**
   * Creates a SeparateChainingHashSet using provided hash table.
   * <p> Time complexity: O(1)
   */
  private SeparateChainingHashSet(SeparateChainingHashTable<T> hastTable) {
    this.hashTable = hastTable;
  }

  /**
   * Creates a new empty SeparateChainingHashSet.
   * <p> Time complexity: O(1)
   *
   * @param numChains Number of separate chains in hash table (should be a prime number).
   * @param maxLoadFactor Maximum load factor for hash table.
   *
   * @throws IllegalArgumentException if numChains is less than 1.
   */
  public SeparateChainingHashSet(int numChains, double maxLoadFactor) {
    this(new SeparateChainingHashTable<>(numChains, maxLoadFactor));
  }

  /**
   * Creates a new empty SeparateChainingHashSet.
   * <p> Time complexity: O(1)
   */
  public SeparateChainingHashSet() {
    this(new SeparateChainingHashTable<>());
  }

  /**
   * Creates a new empty SeparateChainingHashSet.
   * <p> Time complexity: O(1)
   *
   * @return New SeparateChainingHashSet.
   */
  public static <T> SeparateChainingHashSet<T> empty() {
    return new SeparateChainingHashSet<>();
  }

  /**
   * Creates a new empty SeparateChainingHashSet that can accommodate capacity elements without rehashing.
   * <p> Time complexity: O(1)
   *
   * @param capacity Number of elements to accommodate.
   *
   * @return New SeparateChainingHashSet with given capacity.
   *
   * @throws IllegalArgumentException if initial capacity (capacity) is less than 1.
   */
  public static <T> SeparateChainingHashSet<T> withCapacity(int capacity) {
    return new SeparateChainingHashSet<>(SeparateChainingHashTable.withCapacity(capacity));
  }

  /**
   * Creates a new SeparateChainingHashSet with provided elements.
   * <p> Time complexity: near O(n)
   *
   * @param elements Elements to include in new set.
   * @param <T> Type of elements in new set.
   *
   * @return New SeparateChainingHashSet with provided elements.
   */
  @SafeVarargs
  public static <T> SeparateChainingHashSet<T> of(T... elements) {
    SeparateChainingHashSet<T> hashSet = SeparateChainingHashSet.withCapacity(elements.length);
    hashSet.insert(elements);
    return hashSet;
  }

  /**
   * Creates a SeparateChainingHashSet with elements in given iterable.
   * <p> Time complexity: near O(n)
   *
   * @param iterable {@code Iterable} of elements to be added to set.
   * @param <T> Type of elements in iterable.
   *
   * @return a SeparateChainingHashSet with elements in given iterable.
   */
  static <T> SeparateChainingHashSet<T> from(Iterable<T> iterable) {
    SeparateChainingHashSet<T> set = SeparateChainingHashSet.empty();
    for (T element : iterable) {
      set.insert(element);
    }
    return set;
  }

  /**
   * Returns a new SeparateChainingHashSet with same elements as argument.
   * <p> Time complexity: O(n)
   *
   * @param that SeparateChainingHashSet to be copied.
   *
   * @return a new SeparateChainingHashSet with same elements as {@code that}.
   */
  public static <T> SeparateChainingHashSet<T> copyOf(SeparateChainingHashSet<T> that) {
    return new SeparateChainingHashSet<>(SeparateChainingHashTable.copyOf(that.hashTable));
  }

  /**
   * Returns a new SeparateChainingHashSet with same elements as argument.
   * <p> Time complexity: near O(n)
   *
   * @param that Set to be copied.
   *
   * @return a new SeparateChainingHashSet with same elements as {@code that}.
   */
  public static <T> SeparateChainingHashSet<T> copyOf(Set<T> that) {
    if (that instanceof SeparateChainingHashSet<T> hashSet) {
      // use specialized version for SeparateChainingHashSet
      return copyOf(hashSet);
    }
    SeparateChainingHashSet<T> copy = SeparateChainingHashSet.withCapacity(that.size());
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
  public boolean isEmpty() {
    return hashTable.isEmpty();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public int size() {
    return hashTable.size();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: near O(1)
   */
  @Override
  public void insert(T element) {
    hashTable.insert(element);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: near O(1)
   */
  @Override
  public boolean contains(T element) {
    return hashTable.contains(element);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: near O(1)
   */
  @Override
  public void delete(T element) {
    hashTable.delete(element);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public void clear() {
    hashTable.clear();
  }

  /**
   * Iterator over elements in set. Notice that {@code remove} method is not supported. Note also that set should not be
   * modified during iteration as iterator state may become inconsistent.
   *
   * @see java.lang.Iterable#iterator()
   */
  @Override
  public Iterator<T> iterator() {
    return hashTable.iterator();
  }
}
