package org.uma.ed.datastructures.dictionary;


import java.util.Comparator;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * Sorted dictionaries implemented using a sorted linked structure.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class SortedLinkedDictionary<K, V> extends AbstractSortedDictionary<K, V> implements SortedDictionary<K, V> {
  private static final class Node<K, V> {
    Entry<K, V> entry;
    Node<K, V> next;

    Node(Entry<K, V> entry, Node<K, V> next) {
      this.entry = entry;
      this.next = next;
    }
  }

  private Node<K, V> first, last;
  private int size;
  private final Comparator<K> comparator;

  /**
   * Constructs an empty SortedLinkedDictionary. keys are sorted using provided comparator.
   * <p> Time complexity: O(1)
   */
  public SortedLinkedDictionary(Comparator<K> comparator) {
    this.first = null;
    this.last = null;
    this.size = 0;
    this.comparator = comparator;
  }

  /**
   * Constructs an empty SortedLinkedDictionary. keys are sorted using provided comparator.
   * <p> Time complexity: O(1)
   */
  public static <K, V> SortedLinkedDictionary<K, V> empty(Comparator<K> comparator) {
    return new SortedLinkedDictionary<>(comparator);
  }

  /**
   * Constructs an empty SortedLinkedDictionary. keys are sorted using their natural order.
   * <p> Time complexity: O(1)
   */
  public static <K extends Comparable<? super K>, V> SortedLinkedDictionary<K, V> empty() {
    return new SortedLinkedDictionary<K, V>(Comparator.naturalOrder());
  }

  /**
   * Returns a new SortedLinkedDictionary with provided entries and with keys sorted using provided comparator.
   *
   * @param comparator Comparator defining order of keys in new sorted dictionary.
   * @param entries Entries to include in new dictionary.
   * @param <K> Type of keys.
   * @param <V> Type of values.
   *
   * @return a new SortedLinkedDictionary with provided entries and keys sorted using provided comparator.
   */
  @SafeVarargs
  public static <K, V> SortedLinkedDictionary<K, V> of(Comparator<K> comparator, Entry<K, V>... entries) {
    SortedLinkedDictionary<K, V> dictionary = new SortedLinkedDictionary<>(comparator);
    for (Entry<K, V> entry : entries) {
      dictionary.insert(entry);
    }
    return dictionary;
  }

  /**
   * Returns a new SortedLinkedDictionary with provided entries and with keys sorted using their natural order.
   *
   * @param entries Entries to include in new dictionary.
   * @param <K> Type of keys.
   * @param <V> Type of values.
   *
   * @return a new SortedLinkedDictionary with provided entries and keys sorted using their natural order.
   */
  @SafeVarargs
  public static <K extends Comparable<? super K>, V> SortedLinkedDictionary<K, V> of(Entry<K, V>... entries) {
    return of(Comparator.naturalOrder(), entries);
  }

  /**
   * Returns a new SortedLinkedDictionary with provided entries and with keys sorted using provided comparator.
   *
   * @param comparator Comparator defining order of keys in new sorted dictionary.
   * @param entries Entries to include in new dictionary.
   * @param <K> Type of keys.
   * @param <V> Type of values.
   *
   * @return a new SortedLinkedDictionary with provided entries and keys sorted using provided comparator.
   */
  public static <K, V> SortedLinkedDictionary<K, V> from(Comparator<K> comparator, Iterable<Entry<K, V>> entries) {
    SortedLinkedDictionary<K, V> dictionary = new SortedLinkedDictionary<>(comparator);
    for (Entry<K, V> entry : entries) {
      dictionary.insert(entry);
    }
    return dictionary;
  }

  /**
   * Returns a new SortedLinkedDictionary with provided entries and with keys sorted using their natural order.
   *
   * @param entries Entries to include in new dictionary.
   * @param <K> Type of keys.
   * @param <V> Type of values.
   *
   * @return a new SortedLinkedDictionary with provided entries and keys sorted using their natural order.
   */
  public static <K extends Comparable<? super K>, V> SortedLinkedDictionary<K, V> from(Iterable<Entry<K, V>> entries) {
    return from(Comparator.naturalOrder(), entries);
  }

  /**
   * Returns a new SortedLinkedDictionary with same elements as argument.
   * <p> Time complexity: O(n)
   *
   * @param that sorted dictionary to be copied.
   * @param <K> Type of keys.
   * @param <V> Type of values.
   *
   * @return a SortedLinkedDictionary with same keys and values as argument.
   */
  public static <K, V> SortedLinkedDictionary<K, V> copyOf(SortedDictionary<K, V> that) {
    SortedLinkedDictionary<K, V> copy = new SortedLinkedDictionary<>(that.comparator());
    for (Entry<K, V> entry : that.entries()) {
      copy.append(entry);
    }
    return copy;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public boolean isEmpty() {
    return first == null;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public int size() {
    return size;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public Comparator<K> comparator() {
    return comparator;
  }

  private final class Finder {
    Node<K, V> previous, current;
    boolean found;

    public Finder(K key) {
      previous = null;
      current = first;

      int cmp = 0;
      while (current != null && (cmp = comparator.compare(key, current.entry.key())) > 0) {
        previous = current;
        current = current.next;
      }

      found = current != null && cmp == 0;
    }
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(n)
   */
  @Override
  public void insert(Entry<K, V> entry) {
    Finder finder = new Finder(entry.key());

    if (finder.found) {
      finder.current.entry = entry; // update association
    } else {
      if (finder.previous == null) { // Insert in first position
        Node<K, V> node = new Node<>(entry, first);
        first = node;
        if (last == null) {
          last = node;
        }
      } else { // Insert after previous;
        Node<K, V> node = new Node<>(entry, finder.current);
        if (finder.current == null) {
          last = node;
        }
        finder.previous.next = node;
      }
      size++;
    }
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(n)
   */
  @Override
  public V valueOf(K key) {
    Finder finder = new Finder(key);
    return finder.found ? finder.current.entry.value() : null;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(n)
   */
  @Override
  public boolean isDefinedAt(K key) {
    Finder finder = new Finder(key);
    return finder.found;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(n)
   */
  @Override
  public void delete(K key) {
    Finder finder = new Finder(key);
    if (finder.found) {
      if (finder.previous == null) {
        first = finder.current.next;
        if (first == null) {
          last = null;
        }
      } else {
        finder.previous.next = finder.current.next;
        if (finder.current == last) {
          last = finder.previous;
        }
      }
      size--;
    }
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public void clear() {
    first = null;
    last = null;
    size = 0;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public Entry<K, V> minimum() {
    if (first == null) {
      throw new NoSuchElementException("minimum on empty dictionary");
    }
    return first.entry;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public Entry<K, V> maximum() {
    if (first == null) {
      throw new NoSuchElementException("maximum on empty dictionary");
    }
    return last.entry;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public Iterable<K> keys() {
    return KeyIterator::new;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public Iterable<V> values() {
    return ValueIterator::new;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public Iterable<Entry<K, V>> entries() {
    return EntriesIterator::new;
  }

  @Override
  public Iterator<Entry<K, V>> iterator() {
    return entries().iterator();
  }

  private void append(Entry<K, V> entry) {
    assert first == null || comparator.compare(entry.key(), last.entry.key()) > 0;

    Node<K, V> node = new Node<>(entry, null);
    if (first == null) {
      first = node;
    } else {
      last.next = node;
    }
    last = node;
    size++;
  }

  // Almost an iterator on nodes in linked structure
  private class NodeIterator {
    Node<K, V> current;

    public NodeIterator() {
      current = first;
    }

    public boolean hasNext() {
      return current != null;
    }

    public Node<K, V> nextNode() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      Node<K, V> node = current;
      current = current.next;
      return node;
    }
  }

  private final class KeyIterator extends NodeIterator implements Iterator<K> {
    @Override
    public K next() {
      return super.nextNode().entry.key();
    }
  }

  private final class ValueIterator extends NodeIterator implements Iterator<V> {
    @Override
    public V next() {
      return super.nextNode().entry.value();
    }
  }

  private final class EntriesIterator extends NodeIterator implements Iterator<Entry<K, V>> {
    @Override
    public Entry<K, V> next() {
      Node<K, V> node = super.nextNode();
      return node.entry;
    }
  }
}
