package org.uma.ed.datastructures.queue;

/**
 * This class represents a Queue data structure implemented using a linked structure of nodes.
 * Each node in the structure contains a reference to the next node and an element of type T.
 * The structure maintains references to first and last nodes in queue.
 **
 * @param <T> Type of elements in queue.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class LinkedQueue<T> extends AbstractQueue<T> implements Queue<T> {
  /**
   * This class represents a node in a linked structure. Each node contains an element and a reference to the next node.
   * @param <E> Type of elements in node.
   */
  private static final class Node<E> {
    E element;
    Node<E> next;

    Node(E element, Node<E> next) {
      this.element = element;
      this.next = next;
    }
  }

  /*
   * INVARIANT:
   * - if queue is empty, `first` and `last` are null.
   * - if queue is not empty, `first` references first node in queue and `last` references last node in queue.
   * - each node in queue contains a reference to next node or null if it is the last node.
   * - `size` is number of elements in queue.
   */

  /**
   * References to first and last nodes in queue.
   */
  private Node<T> first;
  private Node<T> last;

  /**
   * Number of elements in queue.
   */
  private int size;

  /**
   * Creates an empty LinkedQueue.
   * <p> Time complexity: O(1)
   */
  public LinkedQueue() {
    first = null;
    last = null;
    size = 0;
  }

  /**
   * Creates an empty LinkedQueue.
   * <p> Time complexity: O(1)
   */
  public static <T> LinkedQueue<T> empty() {
    return new LinkedQueue<>();
  }

  /**
   * Creates a LinkedQueue with given elements.
   * <p> Time complexity: O(n)
   *
   * @param elements elements to be added to queue.
   * @param <T> Type of elements in queue.
   *
   * @return a LinkedQueue with given elements.
   */
  @SafeVarargs
  public static <T> LinkedQueue<T> of(T... elements) {
    LinkedQueue<T> queue = new LinkedQueue<>();
    for (T element : elements) {
      queue.enqueue(element);
    }
    return queue;
  }

  /**
   * Creates a LinkedQueue with elements in given iterable.
   * <p> Time complexity: O(n)
   *
   * @param iterable {@code Iterable} of elements to be added to queue.
   * @param <T> Type of elements in iterable.
   *
   * @return a LinkedQueue with elements in given iterable.
   */
  public static <T> LinkedQueue<T> from(Iterable<T> iterable) {
    LinkedQueue<T> queue = new LinkedQueue<>();
    for (T element : iterable) {
      queue.enqueue(element);
    }
    return queue;
  }

  /**
   * Returns a new LinkedQueue with same elements in same order as argument.
   * <p> Time complexity: O(n)
   *
   * @param that LinkedQueue to be copied.
   *
   * @return a new LinkedQueue with same elements and order as {@code that}.
   */
  public static <T> LinkedQueue<T> copyOf(LinkedQueue<T> that) {
    LinkedQueue<T> copy = new LinkedQueue<>();
    int size = that.size();
    if (size > 0) {
      // copy first node
      Node<T> thatNode = that.first;
      Node<T> last = new Node<>(thatNode.element, null);
      copy.first = last;

      // copy rest of nodes
      thatNode = thatNode.next;
      while (thatNode != null) {
        last.next = new Node<>(thatNode.element, null);
        thatNode = thatNode.next;
        last = last.next;
      }
      copy.last = last;
      copy.size = size;
    }
    return copy;
  }

  /**
   * Returns a new LinkedQueue with same elements in same order as argument.
   * <p> Time complexity: O(n)
   *
   * @param that Queue to be copied.
   *
   * @return a new LinkedQueue with same elements and order as {@code that}.
   */
  public static <T> LinkedQueue<T> copyOf(Queue<T> that) {
    if (that instanceof LinkedQueue<T> linkedQueue) {
      // use specialized version for LinkedQueue
      return copyOf(linkedQueue);
    }
    LinkedQueue<T> copy = new LinkedQueue<>();
    int size = that.size();
    if (size > 0) {
      // copy first node
      Node<T> last = new Node<>(that.first(), null);
      that.dequeue();
      copy.first = last;

      // copy rest of nodes
      while (!that.isEmpty()) {
        last.next = new Node<>(that.first(), null);
        that.dequeue();
        last = last.next;
      }
      copy.last = last;
      copy.size = size;

      // restore that elements
      for (Node<T> node = copy.first; node != null; node = node.next) {
        that.enqueue(node.element);
      }
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
   *
   * @throws EmptyQueueException {@inheritDoc}
   */
  @Override
  public T first() {
    if (isEmpty()) {
      throw new EmptyQueueException("first on empty queue");
    }
    return first.element;
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
    first = first.next;
    if (first == null) { // queue became empty
      last = null; // let GC (Garbage Collector) reclaim memory
    }
    size--;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public void enqueue(T element) {
    Node<T> node = new Node<>(element, null);
    if (first == null) { // queue was empty
      first = node;
    } else {
      last.next = node;
    }
    last = node;
    size++;
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
   * Returns a protected iterable over elements in queue.
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
