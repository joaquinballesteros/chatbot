package org.uma.ed.datastructures.set;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * A SortedArraySet is a set data structure that maintains its elements in a sorted order.
 * It implements the SortedSet interface and uses an array to store the elements.
 * The order of elements is defined by the provided comparator or their natural order if no comparator is provided.
 *
 * @param <T> Type of elements in set.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class SortedArraySet<T> extends AbstractSortedSet<T> implements SortedSet<T> {
  /*
   * INVARIANT:
   *  - Elements in the array elements[0..size-1] are in ascending order.
   *  - The array contains no duplicate elements.
   *  - size is number of elements in set.
   */

  /**
   * Default initial array capacity for a SortedArraySet if none is provided.
   */
  private static final int DEFAULT_INITIAL_CAPACITY = 16;

  /**
   * Comparator defining order of elements in this sorted set.
   */
  private final Comparator<T> comparator;

  /**
   * Array storing elements in this set.
   */
  private T[] elements;

  /**
   * Number of elements in this set.
   */
  private int size;

  /**
   * Creates an empty SortedArraySet with provided comparator and initial capacity.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of elements in this sorted set.
   * @param initialCapacity initial capacity for this set.
   */
  @SuppressWarnings("unchecked")
  public SortedArraySet(Comparator<T> comparator, int initialCapacity) {
    if (initialCapacity <= 0) {
      throw new IllegalArgumentException("initial capacity must be greater than 0");
    }
    this.comparator = comparator;
    this.elements = (T[]) new Object[initialCapacity];
    this.size = 0;
  }

  /**
   * Creates an empty SortedArraySet with provided comparator and default initial capacity.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of elements in this sorted set.
   */
  public SortedArraySet(Comparator<T> comparator) {
    this(comparator, DEFAULT_INITIAL_CAPACITY);
  }

  /**
   * Creates an empty SortedArraySet with provided comparator and initial capacity.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of elements in this sorted set.
   * @param initialCapacity initial capacity for this set.
   */
  public static <T> SortedArraySet<T> withCapacity(Comparator<T> comparator, int initialCapacity) {
    return new SortedArraySet<>(comparator, initialCapacity);
  }

  /**
   * Creates an empty SortedArraySet with provided initial capacity and natural order of elements.
   * <p> Time complexity: O(1)
   *
   * @param initialCapacity initial capacity for this set.
   */
  public static <T extends Comparable<? super T>> SortedArraySet<T> withCapacity(int initialCapacity) {
    return new SortedArraySet<T>(Comparator.naturalOrder(), initialCapacity);
  }

  /**
   * Creates an empty SortedArraySet with provided default initial capacity and comparator.
   * <p> Time complexity: O(1)
   * @param comparator Comparator defining order of elements in this sorted set.
   */
  public static <T> SortedArraySet<T> empty(Comparator<T> comparator) {
    return new SortedArraySet<>(comparator, DEFAULT_INITIAL_CAPACITY);
  }

  /**
   * Creates an empty SortedArraySet with provided default initial capacity and natural order of elements.
   * <p> Time complexity: O(1)
   */
  public static <T extends Comparable<? super T>> SortedArraySet<T> empty() {
    return new SortedArraySet<T>(Comparator.naturalOrder(), DEFAULT_INITIAL_CAPACITY);
  }

  /**
   * Returns a new SortedArraySet with given comparator and elements.
   * <p> Time complexity: O(n²)
   *
   * @param comparator Comparator defining order of elements in new sorted set.
   * @param elements Elements to include in new sorted set.
   * @param <T> Type of elements in set.
   *
   * @return a new SortedArraySet with given comparator and elements.
   */
  @SafeVarargs
  public static <T> SortedArraySet<T> of(Comparator<T> comparator, T... elements) {
    SortedArraySet<T> sortedArraySet = new SortedArraySet<>(comparator);
    sortedArraySet.insert(elements);
    return sortedArraySet;
  }

  /**
   * Returns a new SortedArraySet with natural order and provided elements.
   * <p> Time complexity: O(n²)
   *
   * @param elements Elements to include in new sorted set.
   * @param <T> Type of elements in set.
   *
   * @return a new SortedArraySet with natural order and provided elements.
   */
  @SafeVarargs
  public static <T extends Comparable<? super T>> SortedArraySet<T> of(T... elements) {
    return SortedArraySet.of(Comparator.naturalOrder(), elements);
  }

  /**
   * Creates a new SortedArraySet with provided comparator and elements in iterable.
   * <p> Time complexity: O(n²)
   *
   * @param comparator Comparator defining order of elements in new sorted set.
   * @param iterable iterable with elements to include in new sorted set.
   * @param <T> Type of elements in new sorted set.
   *
   * @return New SortedArraySet with provided comparator and elements.
   */
  public static <T> SortedArraySet<T> from(Comparator<T> comparator, Iterable<T> iterable) {
    SortedArraySet<T> sortedArraySet = new SortedArraySet<>(comparator);
    for (T element : iterable) {
      sortedArraySet.insert(element);
    }
    return sortedArraySet;
  }

  /**
   * Creates a new SortedArraySet with natural order and elements in iterable.
   * <p> Time complexity: O(n²)
   *
   * @param iterable iterable with elements to include in new sorted set.
   * @param <T> Type of elements in new sorted set.
   *
   * @return New SortedArraySet with provided comparator and elements.
   */
  public static <T extends Comparable<? super T>> SortedArraySet<T> from(Iterable<T> iterable) {
    return from(Comparator.naturalOrder(), iterable);
  }

  /**
   * Returns a new SortedArraySet with same elements in same order as argument.
   * <p> Time complexity: O(n)
   *
   * @param that Sorted set to be copied.
   *
   * @return a new SortedArraySet with same elements and order as {@code that}.
   */
  public static <T> SortedArraySet<T> copyOf(SortedSet<T> that) {
    SortedArraySet<T> copy = new SortedArraySet<>(that.comparator(), that.size());
    // append elements from sortedSet
    for (T element : that) {
      copy.append(element);
    }
    return copy;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(n)
   */
  @Override
  public void clear() {
    for (int i = 0; i < size; i++) {
      elements[i] = null; // let GC (Garbage Collector) reclaim memory
    }
    size = 0;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public Comparator<T> comparator() {
    return comparator;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public boolean isEmpty() {
    return size == 0;
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
   * Ensures that the capacity of the SortedArraySet is sufficient to hold a new element.
   * If the current capacity is not enough, it is increased by doubling its size.
   */
  private void ensureCapacity() {
    if (size == elements.length) {
      elements = Arrays.copyOf(elements, elements.length * 2);
    }
  }

  /**
   * Finder is a utility class used to locate an element within the sorted array of the SortedArraySet.
   * It performs a binary search to find the position of an element.
   * After a search operation:
   * - If the element is found:
   *   - `found` is set to true.
   *   - `index` is assigned the position of the element within the array.
   * - If the element is not found:
   *   - `found` is set to false.
   *   - `index` indicates the position where the element would be inserted to maintain the sorted order of the array.
   * This class is used to implement the insert, contains, and delete operations in a way that avoids redundant
   * searches.
   */
  private final class Finder {
    boolean found;
    int index;

    /**
     * Constructs a Finder and performs a binary search for the specified element.
     * After construction, the `found` and `index` fields are set according to the outcome of the search.
     *
     * @param element the element to search for
     */
    Finder(T element) {
      found = false;
      int left = 0, right = size - 1, mid = 0;
      while (!found && left <= right) {
        mid = left + (right - left) / 2;
        int cmp = comparator.compare(element, elements[mid]);
        if (cmp == 0) {
          found = true;
        } else if (cmp > 0) {
          left = mid + 1;
        } else {
          right = mid - 1;
        }
      }
      index = found ? mid : left;
    }
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(n)
   */
  @Override
  public void insert(T element) {
    Finder finder = new Finder(element);

    if (!finder.found) {
      // If element is not yet in set
      // Enlarge array if needed
      ensureCapacity();

      // Move elements to right
      for (int i = size; i > finder.index; i--) {
        elements[i] = elements[i - 1];
      }

      // Increase size
      size++;
    }
    // Insert or replace element
    elements[finder.index] = element;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public boolean contains(T element) {
    Finder finder = new Finder(element);
    return finder.found;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(n)
   */
  @Override
  public void delete(T element) {
    Finder finder = new Finder(element);

    // If element is in set
    if (finder.found) {
      // Decrease size
      size--;

      // Move elements to left
      for (int i = finder.index; i < size; i++) {
        elements[i] = elements[i + 1];
      }

      elements[size] = null; // Let GC reclaim memory
    }
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public T minimum() {
    if (isEmpty()) {
      throw new NoSuchElementException("minimum on empty set");
    }
    return elements[0];
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public T maximum() {
    if (isEmpty()) {
      throw new NoSuchElementException("maximum on empty set");
    }
    return elements[size - 1];
  }

  /**
   * Returns an {@code Iterator} for this sorted set. Elements are returned according to set order.
   * Notice that, if the set is structurally modified in any way while the iterator is being used, the iterator state
   * will become inconsistent and will not behave as expected.
   * <p> Time complexity: O(1)
   *
   * @see java.lang.Iterable#iterator()
   *
   * @return an iterator over the elements in this set according to set order.
   */
  @Override
  public Iterator<T> iterator() {
    return new SortedArraySetIterator();
  }

  /**
   * This class represents an iterator for this SortedArraySet that returns elements in set order.
   * INVARIANT:
   *  - The iterator is exhausted when `current` >= `size`.
   *  - `current` indexes the next element to be yielded.
   */
  private final class SortedArraySetIterator implements Iterator<T> {
    int current;

    public SortedArraySetIterator() {
      current = 0;
    }

    public boolean hasNext() {
      return current < size;
    }

    public T next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      T element = elements[current];
      current++;
      return element;
    }
  }

  /**
   * Appends a new element at the end of the SortedArraySet.
   * This method is used to add an element to the set while maintaining the sorted order.
   * PRECONDITION: assumes that the element being added is larger than any existing element in the set.
   * This assumption is based on the precondition that elements are added to the SortedArraySet in ascending order.
   * If the current capacity of the set is not enough to accommodate the new element,
   * the capacity is increased by doubling the size of the underlying array.
   *
   * @param element The element to be added at the end of the set.
   * @throws IllegalArgumentException if the element is smaller than the last element in the set.
   */
  private void append(T element) {
    assert size == 0 || comparator.compare(element, elements[size - 1]) > 0;
    ensureCapacity();
    elements[size] = element;
    size++;
  }

  /**
   * Returns next element in iterator or {@code null} if iterator is exhausted.
   * @param it Iterator to get next element from.
   * @param <T> Type of elements in iterator.
   *
   * @return next element in iterator or {@code null} if iterator is exhausted.
   */
  private static <T> T nextOrNull(Iterator<T> it) {
    return it.hasNext() ? it.next() : null;
  }

  /**
   * Returns a new SortedArraySet that is the difference of the two input sets.
   * PRECONDITION: Both parameters must have same (using {@code ==}) comparators.
   * The difference set contains all elements that are in the first input set but not in the second input set.
   * <p> Time complexity: O(n)
   *
   * @param sortedSet1 The first input set.
   * @param sortedSet2 The second input set.
   * @param <T> Type of elements in both input sets.
   *
   * @return a new SortedArraySet that is the difference of the two input sets.
   */
  public static <T> SortedArraySet<T> difference(SortedSet<T> sortedSet1, SortedSet<T> sortedSet2) {
    /* Should compute a new SortedArraySet including all elements in sortedSet1 which are not in sortedSet2. Neither
     * sortedSet1 nor sortedSet2 should be modified.
     */
    if (sortedSet1.comparator() != sortedSet2.comparator()) {
      throw new IllegalArgumentException("difference: both sorted sets must use same comparator");
    }
    Comparator<T> comparator = sortedSet1.comparator();
    SortedArraySet<T> difference = new SortedArraySet<>(comparator);

    Iterator<T> iterator1 = sortedSet1.iterator();
    Iterator<T> iterator2 = sortedSet2.iterator();

    T element1 = nextOrNull(iterator1);
    T element2 = nextOrNull(iterator2);

    while (element1 != null) {
      if (element2 == null) {
        difference.append(element1);
        element1 = nextOrNull(iterator1);
      } else {
        int cmp = comparator.compare(element1, element2);
        if (cmp < 0) {
          difference.append(element1);
          element1 = nextOrNull(iterator1);
        } else if (cmp == 0) {
          element1 = nextOrNull(iterator1);
          element2 = nextOrNull(iterator2);
        } else {
          element2 = nextOrNull(iterator2);
        }
      }
    }
    return difference;
  }

  /**
   * Returns a new SortedArraySet that is the intersection of the two input sets.
   * PRECONDITION: Both parameters must have same (using {@code ==}) comparators.
   * The intersection set contains all elements that are common to both input sets.
   * <p> Time complexity: O(n min m)
   *
   * @param sortedSet1 The first input set.
   * @param sortedSet2 The second input set.
   * @param <T> Type of elements in both input sets.
   * @return A new SortedArraySet that is the intersection of the two input sets.
   */
  public static <T> SortedArraySet<T> intersection(SortedSet<T> sortedSet1, SortedSet<T> sortedSet2) {
    /* Should compute a new SortedArraySet including only common elements in sortedSet1 and in sortedSet2. Neither
     * sortedSet1 nor sortedSet2 should be modified.
     */
    if (sortedSet1.comparator() != sortedSet2.comparator()) {
      throw new IllegalArgumentException("intersection: both sorted sets must use same comparator");
    }
    Comparator<T> comparator = sortedSet1.comparator();
    SortedArraySet<T> intersection = new SortedArraySet<>(comparator);

    Iterator<T> iterator1 = sortedSet1.iterator();
    Iterator<T> iterator2 = sortedSet2.iterator();

    T element1 = nextOrNull(iterator1);
    T element2 = nextOrNull(iterator2);

    while (element1 != null && element2 != null) {
      int cmp = comparator.compare(element1, element2);
      if (cmp == 0) {
        intersection.append(element1);
        element1 = nextOrNull(iterator1);
        element2 = nextOrNull(iterator2);
      } else if (cmp < 0) {
        element1 = nextOrNull(iterator1);
      } else {
        element2 = nextOrNull(iterator2);
      }
    }
    return intersection;
  }

  /**
   * Returns a new SortedArraySet corresponding to union of two input sets.
   * PRECONDITION: Both parameters must have same (using {@code ==}) comparators.
   * <p> Time complexity: O(n+m)
   *
   * @param sortedSet1 The first input set.
   * @param sortedSet2 The second input set.
   * @param <T> Type of elements in both input sets.
   *
   * @return a new SortedArraySet corresponding to union of the two input sets.
   */
  public static <T> SortedArraySet<T> union(SortedSet<T> sortedSet1, SortedSet<T> sortedSet2) {
    /* Should compute a new SortedArraySet including all elements which are in sortedSet1 or in sortedSet2. Neither
     * sortedSet1 nor sortedSet2 should be modified.
     */
    if (sortedSet1.comparator() != sortedSet2.comparator()) {
      throw new IllegalArgumentException("union: both sorted sets must use same comparator");
    }
    Comparator<T> comparator = sortedSet1.comparator();
    SortedArraySet<T> union = new SortedArraySet<>(comparator);

    Iterator<T> iterator1 = sortedSet1.iterator();
    Iterator<T> iterator2 = sortedSet2.iterator();

    T element1 = nextOrNull(iterator1);
    T element2 = nextOrNull(iterator2);

    while (element1 != null || element2 != null) {
      T element;
      if (element1 == null) {
        element = element2;
        element2 = nextOrNull(iterator2);
      } else if (element2 == null) {
        element = element1;
        element1 = nextOrNull(iterator1);
      } else {
        int cmp = comparator.compare(element1, element2);
        if (cmp == 0) {
          element = element1;
          element1 = nextOrNull(iterator1);
          element2 = nextOrNull(iterator2);
        } else if (cmp < 0) {
          element = element1;
          element1 = nextOrNull(iterator1);
        } else {
          element = element2;
          element2 = nextOrNull(iterator2);
        }
      }
      union.append(element);
    }
    return union;
  }
}
