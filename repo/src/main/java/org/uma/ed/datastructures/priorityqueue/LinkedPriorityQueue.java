package org.uma.ed.datastructures.priorityqueue;

import java.util.Comparator;

/**
 * Priority queue implemented as a sorted linked structure.
 *
 * @param <T> Type of elements.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class LinkedPriorityQueue<T> extends AbstractPriorityQueue<T> implements PriorityQueue<T> {
  private static final class Node<E> {
    E element;
    Node<E> next;

    Node(E element, Node<E> next) {
      this.element = element;
      this.next = next;
    }
  }

  private final Comparator<T> comparator;
  private Node<T> first;
  private int size;

  private LinkedPriorityQueue(Comparator<T> comparator, Node<T> first, int size) {
    this.comparator = comparator;
    this.first = first;
    this.size = size;
  }
  
  /**
   * Creates an empty queue.
   */
  public LinkedPriorityQueue(Comparator<T> comparator) {
    this(comparator, null, 0);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  public Comparator<T> comparator() {
    return comparator;
  }

  public static <T> LinkedPriorityQueue<T> empty(Comparator<T> comparator) {
    return new LinkedPriorityQueue<>(comparator);
  }

  public static <T extends Comparable<? super T>> LinkedPriorityQueue<T> empty() {
    return LinkedPriorityQueue.<T>empty(Comparator.naturalOrder());
  }

  @SafeVarargs
  public static <T> LinkedPriorityQueue<T> of(Comparator<T> comparator, T... elements) {
    LinkedPriorityQueue<T> queue = LinkedPriorityQueue.empty(comparator);
    for(T elem : elements)
      queue.enqueue(elem);
    return queue;
  }

  @SafeVarargs
  public static <T extends Comparable<? super T>> LinkedPriorityQueue<T> of(T... elements) {
    return LinkedPriorityQueue.of(Comparator.naturalOrder(), elements);
  }

  public static <T> LinkedPriorityQueue<T> from(Comparator<T> comparator, Iterable<T> iterable) {
    LinkedPriorityQueue<T> queue = LinkedPriorityQueue.empty(comparator);
    for(T elem : iterable)
      queue.enqueue(elem);
    return queue;
  }

  public static <T extends Comparable<? super T>> LinkedPriorityQueue<T> from(Iterable<T> iterable) {
    return LinkedPriorityQueue.from(Comparator.naturalOrder(), iterable);
  }

  public static <T> LinkedPriorityQueue<T> copyOf(LinkedPriorityQueue<T> queue) {
    return new LinkedPriorityQueue<>(queue.comparator, copyOf(queue.first), queue.size);
  }
  
  private static <T> Node<T> copyOf(Node<T> node) {
    if (node == null) {
      return null;
    } else {
      return new Node<>(node.element, copyOf(node.next));
    }
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
  public void clear() {
    first = null;
    size = 0;
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
   *
   * @throws <code>EmptyPriorityQueueException</code> if queue stores no element.
   */
  @Override
  public void dequeue() {
    if (isEmpty()) {
      throw new EmptyPriorityQueueException("dequeue on empty priority queue");
    }
    first = first.next;
    size--;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   *
   * @throws <code>EmptyPriorityQueueException</code> if queue stores no element.
   */
  @Override
  public T first() {
    if (isEmpty()) {
      throw new EmptyPriorityQueueException("first on empty priority queue");
    }
    return first.element;
  }

  /**
   * {@inheritDoc} Position of new element in queue depends on its priority. The less the value of the element, the
   * higher its priority.
   * <p> Time complexity: O(n)
   */
  @Override
  public void enqueue(T element) {
    Node<T> current = first;
    Node<T> previous = null;
    while ((current != null) && (comparator.compare(element, current.element) >= 0)) {
      // element >= current.element. Advance while element's priority is less or equal to that of
      // element in current node
      previous = current;
      current = current.next;
    }

    if (previous == null) {
      first = new Node<>(element, first);
    } else {
      previous.next = new Node<>(element, current);
    }
    size++;
  }

  /**
   * A protected iterable over elements in this priority queue.
   *
   * @return An iterable over elements in this priority queue.
   */
  protected Iterable<T> elements() {
    return () -> new java.util.Iterator<>() {
      Node<T> current = first;

      public boolean hasNext() {
        return current != null;
      }

      public T next() {
        if (!hasNext()) {
          throw new java.util.NoSuchElementException();
        }
        T element = current.element;
        current = current.next;
        return element;
      }
    };
  }
}
