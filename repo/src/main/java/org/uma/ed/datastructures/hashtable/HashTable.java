package org.uma.ed.datastructures.hashtable;

import java.util.function.Function;
import java.util.function.Predicate;

/**
 * This interface represents a Hash Table data structure that stores unique keys. The uniqueness of keys is ensured by
 * their proper implementation of {@link Object#equals} and {@link Object#hashCode} methods.
 *
 * @param <K> The type of keys in the hash table.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public interface HashTable<K> extends Iterable<K> {
  /**
   * Checks if the hash table is empty.
   *
   * @return {@code true} if the hash table contains no keys, {@code false} otherwise.
   */
  boolean isEmpty();

  /**
   * Retrieves the total number of keys in the hash table.
   *
   * @return The number of keys in the hash table.
   */
  int size();

  /**
   * Inserts a new key into the hash table. If a key that equals the new key is already present in the hash table, the
   * old key is replaced with the new key. Duplicate keys are not supported.
   *
   * @param key The key to be inserted.
   */
  void insert(K key);

  /**
   * Searches for a key in the hash table and returns it if found. If the key is not found, returns {@code null}.
   *
   * @param key The key to search for.
   *
   * @return The key from the hash table if found, {@code null} otherwise.
   */
  K search(K key);

  /**
   * Checks if a key is present in the hash table.
   *
   * @param key The key to check for.
   *
   * @return {@code true} if the key is present in the hash table, {@code false} otherwise.
   */
  boolean contains(K key);

  /**
   * Removes a key from the hash table. If the key is not present, the hash table remains unchanged.
   *
   * @param key The key to be removed.
   */
  void delete(K key);

  /**
   * Removes all keys from the hash table, making it empty.
   */
  void clear();

  /**
   * Performs a delete, update, or insert operation on the hash table based on the provided parameters.
   * - If the key is present and the delete predicate returns true, the key is deleted.
   * - If the key is present and the insert flag is false, the key is updated using the update function.
   * - If the key is not present and the insert flag is true, the key is inserted into the hash table.
   *
   * @param key The key to search for or to insert if not present and the insert flag is true.
   * @param delete The predicate to determine whether to delete the key.
   * @param update The function to update the key.
   * @param insert The flag to determine whether to insert the key if not present.
   */
  void deleteOrUpdateOrInsert(K key, Predicate<K> delete, Function<K, K> update, boolean insert);
}