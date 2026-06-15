package org.uma.ed.datastructures.hashtable;

import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.function.Function;
import java.util.function.Predicate;
import org.uma.ed.datastructures.utils.toString.ToString;

/**
 * Hash tables whose entries are unique keys implemented using open addressing (linear probing). Notice that keys should
 * define {@link Object#equals(Object)} and {@link Object#hashCode} methods properly.
 *
 * @param <K> Type of keys.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class LinearProbingHashTable<K> implements HashTable<K> {
  private static final int DEFAULT_NUM_CELLS = HashPrimes.primeGreaterThan(32);
  private static final double DEFAULT_MAX_LOAD_FACTOR = 0.5;

  private K[] keys; // array to store keys in table
  private int size; // number of keys inserted in table
  private final double maxLoadFactor; // maximum load factor to tolerate

  /**
   * Creates an empty linear probing hash table.
   * <p> Time complexity: O(1)
   *
   * @param numCells Initial number of cells in table (should be a prime number).
   * @param maxLoadFactor Maximum load factor to tolerate. If exceeded, rehashing is performed automatically.
   *
   * @throws IllegalArgumentException if numChains is less than 1.
   */
  @SuppressWarnings("unchecked")
  public LinearProbingHashTable(int numCells, double maxLoadFactor) {
    if (numCells <= 0) {
      throw new IllegalArgumentException("initial number of cells must be greater than 0");
    }
    keys = (K[]) new Object[numCells];
    size = 0;
    this.maxLoadFactor = maxLoadFactor;
  }

  /**
   * Creates an empty linear probing hash table.
   * <p> Time complexity: O(1)
   */
  public LinearProbingHashTable() {
    this(DEFAULT_NUM_CELLS, DEFAULT_MAX_LOAD_FACTOR);
  }

  /**
   * Creates an empty linear probing hash table.
   *
   * @return a new LinearProbingHashTable with given capacity.
   * <p> Time complexity: O(1)
   */
  public static <K> LinearProbingHashTable<K> empty() {
    return new LinearProbingHashTable<>();
  }

  /**
   * Creates an empty linear probing hash table for accommodating size elements so that no rehashing is initially done.
   *
   * @param size number of elements to accommodate.
   *
   * @return a new LinearProbingHashTable with given capacity.
   *
   * @throws IllegalArgumentException if initial capacity (size) is less than 1.
   * <p> Time complexity: O(1)
   */
  public static <K> LinearProbingHashTable<K> withCapacity(int size) {
    if (size <= 0) {
      throw new IllegalArgumentException("initial capacity must be greater than 0");
    }
    return new LinearProbingHashTable<>(HashPrimes.primeGreaterThan((int) (size / DEFAULT_MAX_LOAD_FACTOR)),
        DEFAULT_MAX_LOAD_FACTOR);
  }

  /**
   * Returns a new hash table with same elements as argument.
   * <p> Time complexity: O(n)
   *
   * @param that LinearProbingHashTable to be copied.
   *
   * @return a new LinearProbingHashTable with same elements as {@code that}.
   */
  public static <K> LinearProbingHashTable<K> copyOf(LinearProbingHashTable<K> that) {
    int numCells = that.keys.length;
    LinearProbingHashTable<K> copy = new LinearProbingHashTable<>(numCells, that.maxLoadFactor);
    System.arraycopy(that.keys, 0, copy.keys, 0, numCells);
    copy.size = that.size;
    return copy;
  }

  /**
   * Returns a new hash table with same elements as argument.
   * <p> Time complexity: near O(n)
   *
   * @param that HashTable to be copied.
   *
   * @return a new LinearProbingHashTable with same elements as {@code that}.
   */
  public static <K> LinearProbingHashTable<K> copyOf(HashTable<K> that) {
    if (that instanceof LinearProbingHashTable<K> linearProbingHashTable) {
      // use specialized version for LinearProbingHashTable
      return copyOf(linearProbingHashTable);
    }
    LinearProbingHashTable<K> copy = LinearProbingHashTable.withCapacity(that.size());
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
    return (key.hashCode() & 0x7fffffff) % keys.length;
  }

  // current load factor of hash table
  private double loadFactor() {
    return (double) size / (double) keys.length;
  }

  // circular index increment
  private int advance(int index) {
    return (index + 1) % keys.length;
  }

  // returns index where key is stored or where it should be stored
  private int searchIndex(K key) {
    int index = hash(key);
    while ((keys[index] != null) && (!keys[index].equals(key))) {
      index = advance(index);
    }
    return index;
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

    int index = searchIndex(key);
    if (keys[index] == null) {
      size++;
    }
    keys[index] = key;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: near O(1)
   */
  @Override
  public K search(K key) {
    int index = searchIndex(key);
    return keys[index];
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: near O(1)
   */
  @Override
  public boolean contains(K key) {
    return search(key) != null;
  }

  private void deleteByRelocatingWholeCluster(int index) {
    keys[index] = null; // delete element
    size--;

    // rehash elements after deleted element in same cluster
    index = advance(index);
    while (keys[index] != null) {
      K toRelocate = keys[index];
      keys[index] = null;

      int newIndex = searchIndex(toRelocate);
      keys[newIndex] = toRelocate;

      index = advance(index);
    }
  }

  /**
   * Given a key and its index in table, returns the probe number (0, 1, ...) that reached this index for the key.
   * @param key key in table
   * @param index index where key is stored in table
   *
   * @return probe number that reached such index for key
   */
  private int probeOf(K key, int index) {
    return probeOf(hash(key), index);
  }

  /**
   * Given a hash of a key in table and its index, returns the probe number (0, 1, ...) that reached this index for the key.
   * @param hash hash of key in table
   * @param index index where key is stored in table
   *
   * @return probe number that reached such index for key
   */
  private int probeOf(int hash, int index) {
    int probe = (index - hash) % keys.length;
    if (probe < 0) {
      probe += keys.length; // we need to compute the modulus instead of remainder
    }
    return probe;
  }

  private void delete(int index) {
    K key;
    int keyHash;
    // index is always that of last freed slot
    size--; // we are deleting an element from table
    while (true) {
      keys[index] = null;   // delete element and free slot
      int newIndex = index; // start search from index
      do {
        newIndex = advance(newIndex); // advance to next cell in cluster
        key = keys[newIndex];         // get key at new index
        if (key == null) {            // empty slot found, end of cluster, finish
          return;
        }
        keyHash = hash(key);          // get hash of key
      } while (probeOf(keyHash, index) >= probeOf(keyHash, newIndex)); // while free slot would be probed later for key than its current slot
      keys[index] = key; // we found a key in cluster that should be moved to free slot, move it
      index = newIndex;  // repeat process for new free slot
    }
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: near O(1)
   */
  @Override
  public void delete(K key) {
    int index = searchIndex(key);

    if (keys[index] != null) { // found: delete element
      delete(index);
    }
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: near O(n)
   */
  @Override
  public void clear() {
    Arrays.fill(keys, null);
    size = 0;
  }

  @SuppressWarnings("unchecked")
  private void rehashing() {
    // compute new table size
    int newCapacity = HashPrimes.primeDoubleThan(keys.length);

    K[] oldKeys = keys;

    // allocate new table
    keys = (K[]) new Object[newCapacity];

    // reinsert elements in new table
    for (K oldKey : oldKeys) {
      if (oldKey != null) {
        int newIndex = searchIndex(oldKey); // search for new index in new table
        keys[newIndex] = oldKey; // insert oldKey in new table
      }
    }
  }

  // An iterator on keys stored in hash table
  private final class LinearProbingHashTableIterator implements Iterator<K> {
    int yielded; // number of elements already yielded by this iterator
    int nextIndex; // index of next element to be yielded by this iterator

    public LinearProbingHashTableIterator() {
      yielded = 0;
      nextIndex = -1; // so that after first increment it becomes 0
    }

    // advance nextIndex to index of next to be yielded element
    public void moveForward() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }

      do {
        nextIndex = advance(nextIndex);
      }
      while (keys[nextIndex] == null);

      yielded++;
    }

    public boolean hasNext() {
      return yielded < size;
    }

    public K next() {
      moveForward();
      return keys[nextIndex];
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
    return new LinearProbingHashTableIterator();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: near O(1)
   */
  @Override
  public void deleteOrUpdateOrInsert(K key, Predicate<K> delete, Function<K, K> update, boolean insert) {
    int index = searchIndex(key);
    K oldKey = keys[index];

    if (oldKey != null) { // found
      if (delete.test(oldKey)) {
        // delete element
        delete(index);
      } else { // found. update element
        K newKey = update.apply(oldKey);
        if (oldKey.equals(newKey)) { // new key will have same hash code
          keys[index] = newKey;
        } else { // new key may have different hash code
          delete(index); // delete old key
          insert(newKey); // insert new key at its proper location
        }
      }
    } else if (insert) {
      // key was not present and we have to insert it. We cannot
      // use index as location because rehashing would invalidate such index
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
