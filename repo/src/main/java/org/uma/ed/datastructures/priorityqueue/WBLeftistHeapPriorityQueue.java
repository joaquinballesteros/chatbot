package org.uma.ed.datastructures.priorityqueue;

import java.util.Comparator;
import org.uma.ed.datastructures.heap.WBLeftistHeap;

/**
 * Priority queue implemented with a Weight Biased Leftist Heap.
 *
 * @param <T>
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.   
 */
public class WBLeftistHeapPriorityQueue<T> extends AbstractPriorityQueue<T> implements PriorityQueue<T> {
  private final WBLeftistHeap<T> heap;

  private WBLeftistHeapPriorityQueue(WBLeftistHeap<T> heap) {
    this.heap = heap;
  }
  
  /**
   * Creates an empty queue.
   */
  public WBLeftistHeapPriorityQueue(Comparator<T> comparator) {
    this(WBLeftistHeap.empty(comparator));
  }

  public static <T> WBLeftistHeapPriorityQueue<T> empty(Comparator<T> comparator) {
    return new WBLeftistHeapPriorityQueue<>(comparator);
  }

  public static <T extends Comparable<? super T>> WBLeftistHeapPriorityQueue<T> empty() {
    return WBLeftistHeapPriorityQueue.<T>empty(Comparator.naturalOrder());
  }

  @SafeVarargs
  public static <T> WBLeftistHeapPriorityQueue<T> of(Comparator<T> comparator, T... elements) {
    return new WBLeftistHeapPriorityQueue<>(WBLeftistHeap.of(comparator, elements));
  }

  @SafeVarargs
  public static <T extends Comparable<? super T>> WBLeftistHeapPriorityQueue<T> of(T... elements) {
    return WBLeftistHeapPriorityQueue.of(Comparator.naturalOrder(), elements);
  }

  public static <T> WBLeftistHeapPriorityQueue<T> from(Comparator<T> comparator, Iterable<T> iterable) {
    return new WBLeftistHeapPriorityQueue<>(WBLeftistHeap.from(comparator, iterable));
  }

  public static <T extends Comparable<? super T>> WBLeftistHeapPriorityQueue<T> from(Iterable<T> iterable) {
    return WBLeftistHeapPriorityQueue.from(Comparator.naturalOrder(), iterable);
  }

  public static <T> WBLeftistHeapPriorityQueue<T> copyOf(WBLeftistHeapPriorityQueue<T> queue) {
    return new WBLeftistHeapPriorityQueue<>(WBLeftistHeap.copyOf(queue.heap));
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public Comparator<T> comparator() {
    return heap.comparator();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(n)
   */
  @Override
  public int size() {
    return heap.size();
  }
  
  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public boolean isEmpty() {
    return heap.isEmpty();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public void clear() {
    heap.clear();
  }

  /**
   * {@inheritDoc} Position of new element in queue depends on its priority. The less the value of the element, the
   * higher its priority.
   * <p> Time complexity: O(log n)
   */
  @Override
  public void enqueue(T element) {
    heap.insert(element);
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
    } else {
      return heap.minimum();
    }
  }

  /**
   * {@inheritDoc} Position of new element in queue depends on its priority. The less the value of the element, the
   * higher its priority.
   * <p> Time complexity: O(log n)   *
   * @throws <code>EmptyPriorityQueueException</code> if queue stores no element.
   */
  @Override
  public void dequeue() {
    if (isEmpty()) {
      throw new EmptyPriorityQueueException("dequeue on empty priority queue");
    } else {
      heap.deleteMinimum();
    }
  }

  /**
   * A protected iterable over elements in this priority queue.
   *
   * @return An iterable over elements in this priority queue.
   */
  protected Iterable<T> elements() {
    return () -> new java.util.Iterator<>() {
      private final WBLeftistHeap<T> copy = WBLeftistHeap.copyOf(heap);

      public boolean hasNext() {
        return !copy.isEmpty();
      }

      public T next() {
        if (!hasNext()) {
          throw new java.util.NoSuchElementException();
        }
        T element = copy.minimum();
        copy.deleteMinimum();
        return element;
      }
    };
  }
}
