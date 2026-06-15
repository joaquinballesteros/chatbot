package org.uma.ed.datastructures.dictionary;

import java.util.Comparator;
import java.util.Iterator;
import java.util.NoSuchElementException;

import org.uma.ed.datastructures.searchtree.RBT;
import org.uma.ed.datastructures.searchtree.SearchTree;

/**
 * Dictionaries (finite maps) associating different keys to values implemented as red black  trees sorted by keys. 
 * Notice that the order of keys is provided by the corresponding {@code Comparator}.
 *
 * @param <K> Type of keys.
 * @param <V> Types of values.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class RBTDictionary<K, V> extends AbstractSortedDictionary<K, V> implements SortedDictionary<K, V> {
  private final Comparator<K> comparator;
  private final SearchTree<Entry<K, V>> rbTree;

  private RBTDictionary(Comparator<K> comparator, SearchTree<Entry<K, V>> rbTree) {
    this.comparator = comparator;
    this.rbTree = rbTree;
  }

  /**
   * Creates an empty RBTDictionary. keys are sorted using provided Comparator.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of keys in this sorted dictionary.
   */
  public RBTDictionary(Comparator<K> comparator) {
    this(comparator, new RBT<>(Entry.onKeyComparator(comparator)));
  }

  /**
   * Constructs an empty RBTDictionary. keys are sorted using provided comparator.
   * <p> Time complexity: O(1)
   */
  public static <K, V> RBTDictionary<K, V> empty(Comparator<K> comparator) {
    return new RBTDictionary<>(comparator);
  }

  /**
   * Constructs an empty RBTDictionary. keys are sorted using their natural order.
   * <p> Time complexity: O(1)
   */
  public static <K extends Comparable<? super K>, V> RBTDictionary<K, V> empty() {
    return new RBTDictionary<K, V>(Comparator.naturalOrder());
  }

  /**
   * Returns a new RBTDictionary with provided entries and with keys sorted using provided comparator.
   *
   * @param comparator Comparator defining order of keys in new sorted dictionary.
   * @param entries Entries to include in new dictionary.
   * @param <K> Type of keys.
   * @param <V> Type of values.
   *
   * @return a new RBTDictionary with provided entries and keys sorted using provided comparator.
   */
  @SafeVarargs
  public static <K, V> RBTDictionary<K, V> of(Comparator<K> comparator, Entry<K, V>... entries) {
    RBTDictionary<K, V> dictionary = new RBTDictionary<>(comparator);
    for (Entry<K, V> entry : entries) {
      dictionary.insert(entry);
    }
    return dictionary;
  }

  /**
   * Returns a new RBTDictionary with provided entries and with keys sorted using their natural order.
   *
   * @param entries Entries to include in new dictionary.
   * @param <K> Type of keys.
   * @param <V> Type of values.
   *
   * @return a new RBTDictionary with provided entries and keys sorted using their natural order.
   */
  @SafeVarargs
  public static <K extends Comparable<? super K>, V> RBTDictionary<K, V> of(Entry<K, V>... entries) {
    return of(Comparator.naturalOrder(), entries);
  }

  /**
   * Returns a new RBTDictionary with entries in provided iterable and with keys sorted using provided comparator.
   *
   * @param comparator Comparator defining order of keys in new sorted dictionary.
   * @param iterable iterable with entries to include in new dictionary.
   * @param <K> Type of keys.
   * @param <V> Type of values.
   *
   * @return a new RBTDictionary with provided iterable and keys sorted using provided comparator.
   */
  public static <K, V> RBTDictionary<K, V> from(Comparator<K> comparator, Iterable<Entry<K, V>> iterable) {
    RBTDictionary<K, V> dictionary = new RBTDictionary<>(comparator);
    for (Entry<K, V> entry : iterable) {
      dictionary.insert(entry);
    }
    return dictionary;
  }

  /**
   * Returns a new RBTDictionary with provided entries and with keys sorted using their natural order.
   *
   * @param entries Entries to include in new dictionary.
   * @param <K> Type of keys.
   * @param <V> Type of values.
   *
   * @return a new RBTDictionary with provided entries and keys sorted using their natural order.
   */
  public static <K extends Comparable<? super K>, V> RBTDictionary<K, V> from(Iterable<Entry<K, V>> entries) {
    return from(Comparator.naturalOrder(), entries);
  }

  /**
   * Returns a new RBTDictionary with same elements as argument.
   * <p> Time complexity: O(n)
   *
   * @param that RBTDictionary to be copied.
   *
   * @return a new RBTDictionary with same elements as {@code that}.
   */
  public static <K, V> RBTDictionary<K, V> copyOf(RBTDictionary<K, V> that) {
    return new RBTDictionary<>(that.comparator, RBT.copyOf(that.rbTree));
  }

  /**
   * Returns a new RBTDictionary with same elements as argument.
   * <p> Time complexity: O(n x log n)
   *
   * @param that Dictionary to be copied.
   *
   * @return a new RBTDictionary with same elements as {@code that}.
   */
  public static <K, V> RBTDictionary<K, V> copyOf(SortedDictionary<K, V> that) {
    if (that instanceof RBTDictionary<K, V> rbtDictionary) {
      // use specialized version for RBTDictionary
      return copyOf(rbtDictionary);
    }
    RBTDictionary<K, V> copy = new RBTDictionary<>(that.comparator());
    for (Entry<K, V> entry : that.entries()) {
      copy.insert(entry.key(), entry.value());
    }
    return copy;
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
   * <p> Time complexity: O(1)
   */
  @Override
  public Comparator<K> comparator() {
    return comparator;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void insert(Entry<K, V> entry) {
    rbTree.insert(entry);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public V valueOf(K key) {
    Entry<K, V> entry = rbTree.search(Entry.withKey(key));
    return entry == null ? null : entry.value();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public boolean isDefinedAt(K key) {
    return rbTree.contains(Entry.withKey(key));
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void delete(K key) {
    rbTree.delete(Entry.withKey(key));
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: near O(1)
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
  public Entry<K, V> minimum() {
    if (rbTree.isEmpty()) {
      throw new NoSuchElementException("minimum on empty dictionary");
    }
    return rbTree.minimum();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public Entry<K, V> maximum() {
    if (rbTree.isEmpty()) {
      throw new NoSuchElementException("maximum on empty dictionary");
    }
    return rbTree.maximum();
  }

  private class EntryIterator {
    private final Iterator<Entry<K, V>> iterator;

    private EntryIterator(Iterator<Entry<K, V>> iterator) {
      this.iterator = iterator;
    }

    public boolean hasNext() {
      return iterator.hasNext();
    }

    public Entry<K, V> nextEntry() {
      return iterator.next();
    }
  }

  private final class KeyIterator extends EntryIterator implements Iterator<K> {
    private KeyIterator(Iterator<Entry<K, V>> iterator) {
      super(iterator);
    }

    public K next() {
      return nextEntry().key();
    }
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(n)
   */
  @Override
  public Iterable<K> keys() {
    return () -> new KeyIterator(rbTree.inOrder().iterator());
  }

  private final class ValueIterator extends EntryIterator implements Iterator<V> {
    private ValueIterator(Iterator<Entry<K, V>> iterator) {
      super(iterator);
    }

    public V next() {
      return nextEntry().value();
    }
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(n)
   */
  @Override
  public Iterable<V> values() {
    return () -> new ValueIterator(rbTree.inOrder().iterator());
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(n)
   */
  @Override
  public Iterable<Entry<K, V>> entries() {
    return rbTree.inOrder();
  }

  @Override
  public Iterator<Entry<K, V>> iterator() {
    return entries().iterator();
  }
}
