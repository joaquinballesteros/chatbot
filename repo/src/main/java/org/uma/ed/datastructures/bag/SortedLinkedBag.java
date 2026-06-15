package org.uma.ed.datastructures.bag ;

import java.util.Comparator;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * Bags implemented using a sorted linked structure of nodes. Order of elements is defined by provided comparator or
 * natural order if none is provided.
 *
 * @param <T> Type of elements in bag.
 *
 * @author Pablo López, Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class SortedLinkedBag<T> extends AbstractSortedBag<T> implements SortedBag<T> {
  /**
   * A node in the sorted linked structure containing an element, its number of occurrences and a reference to the next
   * node.
   *
   * @param <E> Type of element stored in this node.
   */
  private static final class Node<E> {
    E element;
    int occurrences;
    Node<E> next;

    Node(E element, int occurrences, Node<E> next) {
      this.element = element;
      this.occurrences = occurrences;
      this.next = next;
    }
  }

  /*
   * INVARIANT:
   * - The linked structure maintains elements in ascending order.
   * - Each node contains a unique element (no two nodes contain the same element) and a reference to the next node or
   *   null if it is the last node.
   * - Nodes must have more than zero occurrences; nodes with zero occurrences are eliminated from the linked structure.
   * - `first` is a reference to the first node in the linked structure or null if bag is empty.
   * - `size` is number of elements stored in this bag.
   */

  /**
   * Comparator defining order of elements in sorted linked structure.
   */
  private final Comparator<T> comparator;

  /**
   * Reference to first node in sorted linked structure or null if bag is empty.
   */
  private Node<T> first;

  /**
   * Reference to last node in sorted linked structure or null if bag is empty.
   */
  private Node<T> last;

  /**
   * Number of elements stored in this bag
   */
  private int size;

  /**
   * Constructs an empty sorted bag with order provided by parameter.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of elements in this sorted bag.
   */
  public SortedLinkedBag(Comparator<T> comparator) {
    this.comparator = comparator;
    this.first = null;
    this.last = null;
    this.size = 0;
  }

  /**
   * Returns a new sorted bag with same elements in same order as argument.
   * <p> Time complexity: O(n)
   *
   * @param that Sorted bag to be copied.
   *
   * @return a new SortedLinkedBag with same elements and order as {@code that}.
   */
  public static <T extends Comparable<? super T>> SortedLinkedBag<T> copyOf(SortedBag<T> that) {
    SortedLinkedBag<T> copy = new SortedLinkedBag<>(that.comparator());
    for (T element : that) {
      copy.append(element);
    }
    return copy;
  }

  /**
   * Constructs an empty sorted bag with order provided by parameter.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of elements in this sorted bag.
   */
  public static <T> SortedLinkedBag<T> empty(Comparator<T> comparator) {
    return new SortedLinkedBag<>(comparator);
  }

  /**
   * Constructs an empty sorted bag with natural order of elements.
   * <p> Time complexity: O(1)
   */
  public static <T extends Comparable<? super T>> SortedLinkedBag<T> empty() {
    return new SortedLinkedBag<T>(Comparator.naturalOrder());
  }

  /**
   * Returns a new sorted bag with given comparator and elements.
   * <p> Time complexity: O(n²)
   *
   * @param comparator Comparator defining order of elements in new sorted bag.
   * @param elements Elements to include in new sorted bag.
   * @param <T> Type of elements in bag.
   *
   * @return a new SortedLinkedBag with given comparator and elements.
   */
  @SafeVarargs
  public static <T> SortedLinkedBag<T> of(Comparator<T> comparator, T... elements) {
    SortedLinkedBag<T> sortedLinkedBag = new SortedLinkedBag<>(comparator);
    sortedLinkedBag.insert(elements);
    return sortedLinkedBag;
  }

  /**
   * Returns a new sorted bag with natural order and provided elements.
   * <p> Time complexity: O(n²)
   *
   * @param elements Elements to include in new sorted bag.
   * @param <T> Type of elements in bag.
   *
   * @return a new SortedLinkedBag with natural order and provided elements.
   */
  @SafeVarargs
  public static <T extends Comparable<? super T>> SortedLinkedBag<T> of(T... elements) {
    return of(Comparator.naturalOrder(), elements);
  }

  /**
   * Returns a new sorted bag with given comparator and elements in provided iterable.
   * <p> Time complexity: O(n²)
   *
   * @param comparator Comparator defining order of iterable in new sorted bag.
   * @param iterable iterable with elements to include in new sorted bag.
   * @param <T> Type of elements in bag.
   *
   * @return a new SortedLinkedBag with given comparator and elements in provided iterable.
   */
  public static <T> SortedLinkedBag<T> from(Comparator<T> comparator, Iterable<T> iterable) {
    SortedLinkedBag<T> sortedLinkedBag = new SortedLinkedBag<>(comparator);
    for (T element : iterable) {
      sortedLinkedBag.insert(element);
    }
    return sortedLinkedBag;
  }

  /**
   * Returns a new sorted bag with natural order and iterable in provided iterable.
   * <p> Time complexity: O(n²)
   *
   * @param iterable iterable with elements to include in new sorted bag.
   * @param <T> Type of elements in bag.
   *
   * @return a new SortedLinkedBag with natural order and provided iterable.
   */
  public static <T extends Comparable<? super T>> SortedLinkedBag<T> from(Iterable<T> iterable) {
    return from(Comparator.naturalOrder(), iterable);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public Comparator<T> comparator() {
    return this.comparator;
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
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public void clear() {
    first = null;
    size = 0;
  }

  /**
   * This class searches for an element within a sorted linked structure.
   * - If the element is found:
   *   - `found` is set to true.
   *   - `current` points to the node containing the element.
   *   - `previous` points to the preceding node, or is null if the element is at the first node.
   * - If the element is not found:
   *   - `found` is set to false.
   *   - `current` points to the node that would follow the element.
   *   - `previous` points to the node that would precede the element, or is null if the element would
   *      be at the first node.
   */
  private final class Finder {
    boolean found;
    Node<T> previous, current;

    Finder(T element) {
      previous = null;
      current = first;

      int cmp = 0;
      while (current != null && (cmp = comparator.compare(element, current.element)) > 0) {
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
  public void insert(T element) {
    Finder finder = new Finder(element);

    if (finder.found) {
      finder.current.occurrences++;
    } else if (finder.previous == null) {
      Node<T> node = new Node<>(element, 1, first);
      first = node;
      if (last == null) {
        last = node;
      }
    } else {
      Node<T> node = new Node<>(element, 1, finder.current);
      if (finder.current == null) {
        last = node;
      }
      finder.previous.next = node;
    }
    size++;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(n)
   */
  @Override
  public void delete(T element) {
    Finder finder = new Finder(element);

    if (finder.found) {
      if (finder.current.occurrences > 1) {
        finder.current.occurrences--;
      } else if (finder.previous == null) {
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
   * <p> Time complexity: O(n)
   */
  @Override
  public int occurrences(T element) {
    Finder finder = new Finder(element);

    return finder.found ? finder.current.occurrences : 0;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public T minimum() {
    if (isEmpty()) {
      throw new NoSuchElementException("minimum on empty bag");
    }
    return first.element;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public T maximum() {
    if (isEmpty()) {
      throw new NoSuchElementException("maximum on empty bag");
    }
    return last.element;
  }

  @Override
  public Iterator<T> iterator() {
    return new BagIterator();
  }

  /** Invariant conditions:
   * - `current` refers to the node holding the next element to be returned, or is null if no elements remain.
   * - `returned` tracks the count of elements already yielded from the current node.
   */
  private final class BagIterator implements Iterator<T> {
    Node<T> current;
    int returned;

    BagIterator() {
      current = first;
      returned = 0;
    }

    public boolean hasNext() {
      return (current != null);
    }

    public T next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      T element = current.element;
      returned++;
      if (returned == current.occurrences) {
        // Maintain invariant
        current = current.next;
        returned = 0;
      }
      return element;
    }
  }

  private void append(T element) {
    assert first == null || comparator.compare(element, last.element) >= 0;

    if (first == null) {
      first = new Node<>(element, 1, null);
      last = first;
    } else if (comparator.compare(element, last.element) == 0) {
      last.occurrences++;
    } else {
      Node<T> node = new Node<>(element, 1, null);
      last.next = node;
      last = node;
    }
    size++;
  }

  // Does union of this and bag. Result is stored on this. bag is not modified
  public void union(Bag<T> bag) {
    for (T x : bag) {
      this.insert(x);
    }
  }

  // We use that the argument is iterated in order
  public void union(SortedLinkedBag<T> that) {
    Node<T> prev = null;
    Node<T> current = first;
    Node<T> thatNode = that.first;

    while (current != null && thatNode != null) {
      T x = current.element;
      T y = thatNode.element;

      int cmp = comparator.compare(x, y);
      if (cmp == 0) {
        // Add y to current node
        current.occurrences += thatNode.occurrences;
        // Advance both lists to next nodes
        prev = current;
        current = current.next;
        thatNode = thatNode.next;
      } else if (cmp < 0) {
        // Advance first list to next node
        prev = current;
        current = current.next;
      } else {
        // Create new node for y and insert it before current
        Node<T> newNode = new Node<>(y, thatNode.occurrences, current);
        if (prev == null) {
          first = newNode;
        } else {
          prev.next = newNode;
        }
        prev = newNode; // Update prev to the new node
        // Advance second list
        thatNode = thatNode.next;
      }
    }

    // Add remaining elements from 'that' bag if any
    while (thatNode != null) {
      Node<T> newNode = new Node<>(thatNode.element, thatNode.occurrences, null);
      if (prev == null) {
        first = newNode;
      } else {
        prev.next = newNode;
      }
      prev = newNode; // Update prev to the new node
      thatNode = thatNode.next;
    }

    // Update 'last' to the last node
    last = prev;
  }


  public void intersection(Bag<T> bag) {
    Node<T> prev = null;
    Node<T> current = first;
    while (current != null) {
      current.occurrences = Math.min(current.occurrences, bag.occurrences(current.element));
      // Advance to next node
      if (current.occurrences <= 0) { // delete this node
        if (prev == null) {
          first = current.next;
          if (first == null) { // The list is now empty
            last = null;
          }
        } else {
          prev.next = current.next;
          if (prev.next == null) { // We've deleted the last node
            last = prev;
          }
        }
      } else {
        prev = current;
      }
      current = current.next;
    }
  }


  public void intersection(SortedLinkedBag<T> that) {
    Node<T> prev = null;
    Node<T> current = first;
    Node<T> thatNode = that.first;

    while (current != null && thatNode != null) {
      T x = current.element;
      T y = thatNode.element;

      int cmp = comparator.compare(x, y);
      if (cmp == 0) {
        // Set occurrences to the minimum of both nodes
        current.occurrences = Math.min(current.occurrences, thatNode.occurrences);
        prev = current;
        current = current.next;
        thatNode = thatNode.next;
      } else if (cmp < 0) {
        // Remove x node from result (first list)
        if (prev == null) {
          first = current.next;
        } else {
          prev.next = current.next;
        }
        if (current.next == null) {
          last = prev; // Update last if we deleted the last node
        }
        current = current.next;
      } else {
        // Advance second list
        thatNode = thatNode.next;
      }
    }

    // If first list is not exhausted, then remove remaining elements
    if (current != null) {
      if (prev == null) {
        first = null;
        last = null; // Update last as the list is now empty
      } else {
        prev.next = null;
        last = prev; // Update last to the new end of the list
      }
    }
  }

  public void difference(Bag<T> bag) {
    Node<T> prev = null;
    Node<T> current = first;
    while (current != null) {
      current.occurrences -= bag.occurrences(current.element);
      Node<T> nextNode = current.next; // Save the next node before potentially deleting the current node
      // Advance to next node
      if (current.occurrences <= 0) { // delete this node
        if (prev == null) {
          first = nextNode;
        } else {
          prev.next = nextNode;
        }
        // If we deleted the last node, update the 'last' reference
        if (nextNode == null) {
          last = prev;
        }
      } else {
        prev = current;
      }
      current = nextNode;
    }

    // If the first node is null, the bag is empty, so update 'last' to null
    if (first == null) {
      last = null;
    }
  }

  // We use that argument iterates in order
  public void difference(SortedLinkedBag<T> that) {
    Node<T> prev = null;
    Node<T> current = first;
    Node<T> thatNode = that.first;

    while (current != null && thatNode != null) {
      T x = current.element;
      T y = thatNode.element;

      int cmp = comparator.compare(x, y);
      if (cmp == 0) {
        // Subtract occurrences of y from x
        current.occurrences -= thatNode.occurrences;
        // Advance both lists to next nodes
        if (current.occurrences <= 0) { // delete this node
          if (prev == null) {
            first = current.next;
          } else {
            prev.next = current.next;
          }
          // If we deleted the last node, update the 'last' reference
          if (current.next == null) {
            last = prev;
          }
        } else {
          prev = current;
        }

        current = current.next;
        thatNode = thatNode.next;
      } else if (cmp < 0) {
        // Advance first list to next node
        prev = current;
        current = current.next;
      } else {
        // Advance second list
        thatNode = thatNode.next;
      }
    }

    // If the first node is null, the bag is empty, so update 'last' to null
    if (first == null) {
      last = null;
    }
  }
}
