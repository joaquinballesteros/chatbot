package org.uma.ed.datastructures.hashtable;

import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.function.Function;
import java.util.function.Predicate;
import org.uma.ed.datastructures.utils.toString.ToString;

/**
 * Hash tables whose entries are unique keys implemented using separate chaining. Notice that keys should define
 * {@link Object#equals(Object)} and {@link Object#hashCode} methods properly.
 *
 * @param <K> Type of keys.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class SeparateChainingHashTable<K> implements HashTable<K> {
  /**
   * This class represents a node in a linked list.
   *
   * @param <K> Type of key.
   */
  private static final class Node<K> {
    K key;        // key stored in node
    Node<K> next; // reference to next node in list

    Node(K key, Node<K> next) {
      this.key = key;
      this.next = next;
    }
  }

  private static final int DEFAULT_NUM_CHAINS = HashPrimes.primeGreaterThan(32);
  private static final double DEFAULT_MAX_LOAD_FACTOR = 5;

  private Node<K>[] table; // array of chains
  private int size;        // number of keys inserted in table
  private final double maxLoadFactor; // maximum load factor to tolerate

  /**
   * Creates an empty separate chaining hash table.
   * <p> Time complexity: O(1)
   *
   * @param numChains Number of separate chains (linked lists). Should be a prime number.
   * @param maxLoadFactor Maximum load factor to tolerate. If exceeded, rehashing is performed automatically.
   *
   * @throws IllegalArgumentException if numChains is less than 1.
   */
  @SuppressWarnings("unchecked")
  public SeparateChainingHashTable(int numChains, double maxLoadFactor) {
    if (numChains <= 0) {
      throw new IllegalArgumentException("initial number of chains must be greater than 0");
    }
    this.table = (Node<K>[]) new Node[numChains];
    this.size = 0;
    this.maxLoadFactor = maxLoadFactor;
  }

  /**
   * Creates an empty separate chaining hash table.
   * <p> Time complexity: O(1)
   */
  public SeparateChainingHashTable() {
    this(DEFAULT_NUM_CHAINS, DEFAULT_MAX_LOAD_FACTOR);
  }

  /**
   * Creates an empty separate chaining hash table.
   *
   * @return a new SeparateChainingHashTable with given capacity.
   * <p> Time complexity: O(1)
   */
  public static <K> SeparateChainingHashTable<K> empty() {
    return new SeparateChainingHashTable<>();
  }

  /**
   * Creates an empty separate chaining hash table for accommodating size elements so that no rehashing is initially
   * done.
   *
   * @param size number of elements to accommodate.
   *
   * @return a new SeparateChainingHashTable with given capacity.
   *
   * @throws IllegalArgumentException if initial capacity (size) is less than 1.
   * <p> Time complexity: O(1)
   */
  public static <K> SeparateChainingHashTable<K> withCapacity(int size) {
    if (size <= 0) {
      throw new IllegalArgumentException("initial capacity must be greater than 0");
    }
    return new SeparateChainingHashTable<>(HashPrimes.primeGreaterThan((int) (size / DEFAULT_MAX_LOAD_FACTOR)),
        DEFAULT_MAX_LOAD_FACTOR);
  }

  /**
   * Returns a new hash table with same elements as argument.
   * <p> Time complexity: O(n)
   *
   * @param that SeparateChainingHashTable to be copied.
   *
   * @return a new SeparateChainingHashTable with same elements as {@code that}.
   */
  public static <K> SeparateChainingHashTable<K> copyOf(SeparateChainingHashTable<K> that) {
    int numChains = that.table.length;
    SeparateChainingHashTable<K> copy = new SeparateChainingHashTable<>(numChains, that.maxLoadFactor);
    for (int index = 0; index < numChains; index++) {
      Node<K> thatNode = that.table[index];
      if (thatNode != null) {
        // copy first thatNode
        copy.table[index] = new Node<>(thatNode.key, null);
        Node<K> last = copy.table[index];
        thatNode = thatNode.next;
        // copy remaining nodes
        while (thatNode != null) {
          last.next = new Node<>(thatNode.key, null);
          thatNode = thatNode.next;
          last = last.next;
        }
      }
    }
    copy.size = that.size;
    return copy;
  }

  /**
   * Returns a new hash table with same elements as argument.
   * <p> Time complexity: O(n)
   *
   * @param that HashTable to be copied.
   *
   * @return a new SeparateChainingHashTable with same elements as {@code that}.
   */
  public static <K> SeparateChainingHashTable<K> copyOf(HashTable<K> that) {
    if (that instanceof SeparateChainingHashTable<K> separateChainingHashTable) {
      // use specialized version for SeparateChainingHashTable
      return copyOf(separateChainingHashTable);
    }
    SeparateChainingHashTable<K> copy = SeparateChainingHashTable.withCapacity(that.size());
    for (K key : that) {
      copy.insert(key);
    }
    return copy;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: near O(1)
   */
  @Override
  public boolean isEmpty() {
    return size == 0;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: near O(1)
   */
  @Override
  public int size() {
    return size;
  }

  // hash function for keys
  private int hash(K key) {
    return (key.hashCode() & 0x7fffffff) % table.length;
  }

  // current load factor of hash table
  private double loadFactor() {
    return (double) size / (double) table.length;
  }

  /**
   * Searches for a key in table and returns information about key location:
   * current will be a reference to node containing key or null if key is not in table,
   * previous will be a reference to previous node in chain or null if key is first in chain,
   * index will be the index of chain where key is or should be.
   */
  private final class Finder {
    int index;
    Node<K> previous, current;

    Finder(K key) {
      index = hash(key);
      previous = null;
      current = table[index];

      while ((current != null) && (!current.key.equals(key))) {
        previous = current;
        current = current.next;
      }
    }
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: near O(1)
   */
  @Override
  public void insert(K key) {
    if (loadFactor() > maxLoadFactor) {
      rehashing();
    }

    Finder finder = new Finder(key);
    if (finder.current == null) {
      // key was not present: insert it
      table[finder.index] = new Node<>(key, table[finder.index]);
      size++;
    } else {
      // key was present: update it
      finder.current.key = key;
    }
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: near O(1)
   */
  @Override
  public K search(K key) {
    Finder finder = new Finder(key);
    return finder.current == null ? null : finder.current.key;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: near O(1)
   */
  @Override
  public boolean contains(K key) {
    return search(key) != null;
  }

  private void delete(Finder finder) {
    // delete element
    if (finder.previous == null) {
      // remove first node in chain
      table[finder.index] = finder.current.next;
    } else {
      // remove internal node
      finder.previous.next = finder.current.next;
    }
    size--;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: near O(1)
   */
  @Override
  public void delete(K key) {
    Finder finder = new Finder(key);

    if (finder.current != null) { // key found: delete it
      delete(finder);
    }
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public void clear() {
    Arrays.fill(table, null);
    size = 0;
  }

  @SuppressWarnings("unchecked")
  private void rehashing() {
    // compute new table size
    int newCapacity = HashPrimes.primeDoubleThan(table.length);

    Node<K>[] oldTable = table;

    // allocate new table
    table = (Node<K>[]) new Node[newCapacity];

    for (Node<K> chain : oldTable) {  // for each chain in old table
      Node<K> current = chain;
      while (current != null) { // for each node in chain
        Node<K> node = current;
        current = current.next; // for next iteration of while loop
        // insert node in new table
        int index = hash(node.key);
        node.next = table[index];
        table[index] = node;
      }
    }
  }

  private final class SeparateChainingHashTableIterator implements Iterator<K> {
    int index; // index of chain being traversed
    Node<K> current; // current node in chain being traversed

    public SeparateChainingHashTableIterator() {
      index = 0;
      current = table[index];

      // locate first element
      advance();
    }

    private void advance() {
      while ((current == null) && (index < table.length - 1)) {
        index++;
        current = table[index];
      }
    }

    public boolean hasNext() {
      return (current != null);
    }

    public K next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }

      K key = current.key;
      // advance for next invocation
      current = current.next;
      advance();

      return key;
    }
  }

  /**
   * Iterator over elements in set. Notice that {@code remove} method is not supported. Note also that hash table should
   * not be modified during iteration as iterator state may become inconsistent.
   *
   * @see java.lang.Iterable#iterator()
   */
  @Override
  public Iterator<K> iterator() {
    return new SeparateChainingHashTableIterator();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: near O(1)
   */
  @Override
  public void deleteOrUpdateOrInsert(K key, Predicate<K> delete, Function<K, K> update, boolean insert) {
    Finder finder = new Finder(key);

    if (finder.current != null) { // found
      K oldKey = finder.current.key;
      if (delete.test(oldKey)) {
        // delete element
        delete(finder);
      } else { // found. update element
        K newKey = update.apply(oldKey);
        if (oldKey.equals(newKey)) { // new key will have same hash code
          finder.current.key = newKey;
        } else { // new key may have different hash code
          delete(finder); // delete old key
          insert(newKey); // insert new key at its proper location
        }
      }
    } else if (insert) {
      // key was not present and we have to insert it. We cannot
      // use index as location because a rehashing would invalidate such index
      insert(key);
    }
  }

  /**
   * Returns representation of hash table as a String.
   */
  @Override
  public String toString() {
    return ToString.toString(this);
  }
}
