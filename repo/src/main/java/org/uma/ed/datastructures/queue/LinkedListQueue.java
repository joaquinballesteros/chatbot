package org.uma.ed.datastructures.queue;

import org.uma.ed.datastructures.list.LinkedList;
import org.uma.ed.datastructures.list.List;

/**
 * This class represents a Queue data structure implemented using a linked list of elements.
 * The first element in the queue is represented by the first element in the list.
 *
 * @param <T> Type of elements in queue.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class LinkedListQueue<T> extends AbstractQueue<T> implements Queue<T> {
  /**
   * List of elements in queue.
   */
  private final List<T> elements;

  /* INVARIANT:
   *  - elements contains elements in queue in first to last order.
   *  - size is number of elements in queue.
   */

  /**
   * Creates an empty LinkedListQueue.
   * <p> Time complexity: O(1)
   */
  public LinkedListQueue() {
    elements = LinkedList.empty();
  }

  /**
   * Creates an empty LinkedListQueue.
   * <p> Time complexity: O(1)
   */
  public static <T> LinkedQueue<T> empty() {
    return new LinkedQueue<>();
  }

  /**
   * Creates a LinkedListQueue with given elements.
   * <p> Time complexity: O(n)
   *
   * @param elements elements to be added to queue.
   * @param <T> Type of elements in queue.
   *
   * @return a LinkedListQueue with given elements.
   */
  @SafeVarargs
  public static <T> LinkedListQueue<T> of(T... elements) {
    LinkedListQueue<T> queue = new LinkedListQueue<>();
    for (T element : elements) {
      queue.enqueue(element);
    }
    return queue;
  }

  /**
   * Creates a LinkedListQueue with elements in given iterable.
   * <p> Time complexity: O(n)
   *
   * @param iterable {@code Iterable} of elements to be added to queue.
   * @param <T> Type of elements in iterable.
   *
   * @return a LinkedListQueue with elements in given iterable.
   */
  public static <T> LinkedListQueue<T> from(Iterable<T> iterable) {
    LinkedListQueue<T> queue = new LinkedListQueue<>();
    for (T element : iterable) {
      queue.enqueue(element);
    }
    return queue;
  }

  /**
   * Returns a new LinkedListQueue with same elements in same order as argument.
   * <p> Time complexity: O(n)
   *
   * @param that Queue to be copied.
   *
   * @return a new LinkedListQueue with same elements and order as {@code that}.
   */
  public static <T> LinkedListQueue<T> copyOf(Queue<T> that) {
    if (that instanceof LinkedListQueue<T> linkedListQueue) {
      // use specialized version for LinkedListQueue
      return copyOf(linkedListQueue);
    }
    LinkedListQueue<T> copy = new LinkedListQueue<>();
    while (!that.isEmpty()) {
      copy.enqueue(that.first());
      that.dequeue();
    }
    for (T element : copy.elements) {
      that.enqueue(element);
    }
    return copy;
  }


  /**
   * Returns a new LinkedListQueue with same elements in same order as argument.
   * <p> Time complexity: O(n)
   *
   * @param that LinkedListQueue to be copied.
   *
   * @return a new LinkedListQueue with same elements and order as {@code that}.
   */
  public static <T> LinkedListQueue<T> copyOf(LinkedListQueue<T> that) {
    LinkedListQueue<T> copy = new LinkedListQueue<>();
    for (T element : that.elements) {
      copy.enqueue(element);
    }
    return copy;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public boolean isEmpty() {
    return elements.isEmpty();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public int size() {
    return elements.size();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public void enqueue(T element) {
    elements.append(element);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   *
   * @throws EmptyQueueException {@inheritDoc}
   */
  @Override
  public T first() {
    if (isEmpty()) {
      throw new EmptyQueueException("first on empty queue");
    }
    return elements.get(0);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   *
   * @throws EmptyQueueException {@inheritDoc}
   */
  @Override
  public void dequeue() {
    if (isEmpty()) {
      throw new EmptyQueueException("dequeue on empty queue");
    }
    elements.delete(0);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public void clear() {
    elements.clear();
  }

  /**
   * Returns a protected iterable over elements in queue.
   */
  protected Iterable<T> elements() {
    return elements;
  }
}
