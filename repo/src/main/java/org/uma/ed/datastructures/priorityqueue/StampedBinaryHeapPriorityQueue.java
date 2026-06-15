package org.uma.ed.datastructures.priorityqueue;

import java.util.Comparator;
import org.uma.ed.datastructures.heap.BinaryHeap;

/**
 * A priority queue respecting order of insertions by means of time stamped nodes.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class StampedBinaryHeapPriorityQueue<T> extends AbstractPriorityQueue<T> implements PriorityQueue<T> {
  private final Comparator<T> comparator;
  
  private final class Node implements Comparable<Node> {
    T element;
    long timeStamp;

    static long nextTimeStamp = 0;

    Node(T element) {
      this.element = element;
      this.timeStamp = nextTimeStamp;
      nextTimeStamp++;
    }

    @Override
    public int compareTo(Node that) {
      // first compare elements
      int cmp = comparator.compare(element, that.element);
      // if they are the same, use time stamps
      if (cmp == 0) {
        cmp = Long.compare(timeStamp, that.timeStamp);
      }
      return cmp;
    }
  }

  private final BinaryHeap<Node> heap;

  private StampedBinaryHeapPriorityQueue(Comparator<T> comparator, BinaryHeap<Node> heap) {
    this.comparator = comparator;
    this.heap = heap;
  }

  private StampedBinaryHeapPriorityQueue(Comparator<T> comparator, int initialCapacity) {
    this(comparator, BinaryHeap.withCapacity(Node::compareTo, initialCapacity));
  }

  public StampedBinaryHeapPriorityQueue(Comparator<T> comparator) {
    this(comparator, BinaryHeap.empty());
  }

  public static <T> StampedBinaryHeapPriorityQueue<T> empty(Comparator<T> comparator) {
    return new StampedBinaryHeapPriorityQueue<>(comparator);
  }

  public static <T extends Comparable<? super T>> StampedBinaryHeapPriorityQueue<T> empty() {
    return StampedBinaryHeapPriorityQueue.<T>empty(Comparator.naturalOrder());
  }

  public static <T> StampedBinaryHeapPriorityQueue<T> withCapacity(Comparator<T> comparator, int initialCapacity) {
    return new StampedBinaryHeapPriorityQueue<>(comparator,  initialCapacity);
  }

  public static <T extends Comparable<? super T>> StampedBinaryHeapPriorityQueue<T> withCapacity(int initialCapacity) {
    return StampedBinaryHeapPriorityQueue.<T>withCapacity(Comparator.naturalOrder(), initialCapacity);
  }

  @SafeVarargs
  public static <T> StampedBinaryHeapPriorityQueue<T> of(Comparator<T> comparator, T... elements) {
    StampedBinaryHeapPriorityQueue<T> queue = StampedBinaryHeapPriorityQueue.withCapacity(comparator, elements.length);
    for(T elem : elements)
      queue.enqueue(elem);
    return queue;
  }

  @SafeVarargs
  public static <T extends Comparable<? super T>> StampedBinaryHeapPriorityQueue<T> of(T... elements) {
    return StampedBinaryHeapPriorityQueue.of(Comparator.naturalOrder(), elements);
  }

  public static <T> StampedBinaryHeapPriorityQueue<T> from(Comparator<T> comparator, Iterable<T> iterable) {
    StampedBinaryHeapPriorityQueue<T> queue = StampedBinaryHeapPriorityQueue.empty(comparator);
    for(T elem : iterable)
      queue.enqueue(elem);
    return queue;
  }

  public static <T extends Comparable<? super T>> StampedBinaryHeapPriorityQueue<T> from(Iterable<T> iterable) {
    return StampedBinaryHeapPriorityQueue.from(Comparator.naturalOrder(), iterable);
  }

  public static <T> StampedBinaryHeapPriorityQueue<T> copyOf(StampedBinaryHeapPriorityQueue<T> queue) {
    return new StampedBinaryHeapPriorityQueue<T>(queue.comparator, BinaryHeap.copyOf(queue.heap));
  }
  
  @Override
  public Comparator<T> comparator() {
    return comparator;
  }

  @Override
  public boolean isEmpty() {
    return heap.isEmpty();
  }

  @Override
  public void clear() {
    heap.clear();
  }

  @Override
  public int size() {
    return heap.size();
  }
  
  @Override
  public void enqueue(T element) {
    Node node = new Node(element);
    heap.insert(node);
  }

  @Override
  public T first() {
    if (isEmpty()) {
      throw new EmptyPriorityQueueException("first on empty priority queue");
    }
    return heap.minimum().element;
  }

  @Override
  public void dequeue() {
    if (isEmpty()) {
      throw new EmptyPriorityQueueException("dequeue on empty priority queue");
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
      private final BinaryHeap<Node> copy = BinaryHeap.copyOf(heap);

      public boolean hasNext() {
        return !copy.isEmpty();
      }

      public T next() {
        if (!hasNext()) {
          throw new java.util.NoSuchElementException();
        }
        T element = copy.minimum().element;
        copy.deleteMinimum();
        return element;
      }
    };
  }
}
