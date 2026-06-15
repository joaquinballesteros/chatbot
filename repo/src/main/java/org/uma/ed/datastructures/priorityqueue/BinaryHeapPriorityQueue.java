package org.uma.ed.datastructures.priorityqueue;

import java.util.Comparator;
import org.uma.ed.datastructures.heap.BinaryHeap;

/**
 * Priority queue with a Binary Heap.
 *
 * @param <T> Type of elements
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class BinaryHeapPriorityQueue<T> extends AbstractPriorityQueue<T> implements PriorityQueue<T> {
  private final BinaryHeap<T> heap;

  private BinaryHeapPriorityQueue(BinaryHeap<T> heap) {
    this.heap = heap;
  }

  /**
   * Creates an empty queue.
   * <p> Time complexity: O(1)
   */
  public BinaryHeapPriorityQueue(Comparator<T> comparator) {
    this(BinaryHeap.empty(comparator));
  }

  public static <T> BinaryHeapPriorityQueue<T> empty(Comparator<T> comparator) {
    return new BinaryHeapPriorityQueue<>(comparator);
  }

  public static <T extends Comparable<? super T>> BinaryHeapPriorityQueue<T> empty() {
    return BinaryHeapPriorityQueue.<T>empty(Comparator.naturalOrder());
  }

  public static <T> BinaryHeapPriorityQueue<T> withCapacity(Comparator<T> comparator, int initialCapacity) {
    return new BinaryHeapPriorityQueue<>(BinaryHeap.withCapacity(comparator, initialCapacity));
  }

  public static <T extends Comparable<? super T>> BinaryHeapPriorityQueue<T> withCapacity(int initialCapacity) {
    return BinaryHeapPriorityQueue.<T>withCapacity(Comparator.naturalOrder(), initialCapacity);
  }

  @SafeVarargs
  public static <T> BinaryHeapPriorityQueue<T> of(Comparator<T> comparator, T... elements) {
    return new BinaryHeapPriorityQueue<>(BinaryHeap.of(comparator, elements));
  }

  @SafeVarargs
  public static <T extends Comparable<? super T>> BinaryHeapPriorityQueue<T> of(T... elements) {
    return BinaryHeapPriorityQueue.of(Comparator.naturalOrder(), elements);
  }

  public static <T> BinaryHeapPriorityQueue<T> from(Comparator<T> comparator, Iterable<T> iterable) {
    return new BinaryHeapPriorityQueue<>(BinaryHeap.from(comparator, iterable));
  }

  public static <T extends Comparable<? super T>> BinaryHeapPriorityQueue<T> from(Iterable<T> iterable) {
    return BinaryHeapPriorityQueue.from(Comparator.naturalOrder(), iterable);
  }

  public static <T> BinaryHeapPriorityQueue<T> copyOf(BinaryHeapPriorityQueue<T> queue) {
    return new BinaryHeapPriorityQueue<>(BinaryHeap.copyOf(queue.heap));
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
   * <p> Time complexity: O(1)
   */
  @Override
  public boolean isEmpty() {
    return heap.isEmpty();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(n)
   */
  @Override
  public void clear() {
    heap.clear();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public int size() {
    return heap.size();
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
    }
    return heap.minimum();
  }

  /**
   * {@inheritDoc} Position of new element in queue depends on its priority. The less the value of the element, the
   * higher its priority.
   * <p> Time complexity: O(log n)
   *
   * @throws <code>EmptyPriorityQueueException</code> if queue stores no element.
   */
  @Override
  public void dequeue() {
    if (isEmpty()) {
      throw new EmptyPriorityQueueException("first on empty priority queue");
    }
    heap.deleteMinimum();
  }

  /**
   * A protected iterable over elements in this priority queue.
   *
   * @return An iterable over elements in this priority queue.
   */
  protected Iterable<T> elements() {
    return () -> new java.util.Iterator<>() {
      private final BinaryHeap<T> copy = BinaryHeap.copyOf(heap);

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
