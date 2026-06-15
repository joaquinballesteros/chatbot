package org.uma.ed.datastructures.bag ;

import java.util.Comparator;
import java.util.Iterator;
import java.util.NoSuchElementException;
import org.uma.ed.datastructures.searchtree.RBT;
import org.uma.ed.datastructures.searchtree.SearchTree;

/**
 * Bags implemented using red black trees. Order of elements is defined by provided comparator or natural order if none 
 * is provided.
 *
 * @param <T> Type of elements in set.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class RBTBag<T> extends AbstractSortedBag<T> implements SortedBag<T> {
  private record Pair<E>(E element, int occurrences) {
    static <E> Pair<E> of(E element, int occurrences) {
      return new Pair<>(element, occurrences);
    }

    static <E> Pair<E> withElement(E element) {
      return new Pair<>(element, 0);
    }

    static <E> Comparator<Pair<E>> byElement(Comparator<E> comparator) {
      return (p1, p2) -> comparator.compare(p1.element, p2.element);
    }
  }

  private final Comparator<T> comparator;
  private final SearchTree<Pair<T>> rbTree;

  private RBTBag(Comparator<T> comparator, RBT<Pair<T>> rbTree) {
    this.comparator = comparator;
    this.rbTree = rbTree;
  }

  /**
   * Constructs an empty sorted bag with order provided by parameter.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of elements in this sorted bag.
   */
  public RBTBag(Comparator<T> comparator) {
    this(comparator, RBT.empty(Pair.byElement(comparator)));
  }

  /**
   * Constructs an empty sorted bag with natural order of elements.
   * <p> Time complexity: O(1)
   */
  public static <T extends Comparable<? super T>> RBTBag<T> empty() {
    return new RBTBag<T>(Comparator.naturalOrder());
  }

  /**
   * Constructs an empty sorted bag with order provided by parameter.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of elements in this sorted bag.
   */
  public static <T> RBTBag<T> empty(Comparator<T> comparator) {
    return new RBTBag<>(comparator);
  }

  /**
   * Returns a new RBTBag with same elements as argument.
   * <p> Time complexity: O(n)
   *
   * @param that RBTBag to be copied.
   *
   * @return a new RBTBag with same elements as {@code that}.
   */
  public static <T> RBTBag<T> copyOf(RBTBag<T> that) {
    return new RBTBag<>(that.comparator, RBT.copyOf(that.rbTree));
  }

  /**
   * Returns a new RBTBag with same elements as argument.
   * <p> Time complexity: O(n²)
   *
   * @param that Sorted bag to be copied.
   *
   * @return a new RBTBag with same elements as {@code that}.
   */
  public static <T> RBTBag<T> copyOf(SortedBag<T> that) {
    if (that instanceof RBTBag<T> rbtBag) {
      // use specialized version for RBTBag
      return copyOf(rbtBag);
    }
    // todo could be improved as elements in that are already sorted
    RBTBag<T> copy = new RBTBag<>(that.comparator());
    for (T element : that) {
      copy.insert(element);
    }
    return copy;
  }

  /**
   * Creates a new RBTBag with provided comparator and elements.
   * <p> Time complexity: O(n²)
   *
   * @param comparator Comparator defining order of elements in new sorted bag.
   * @param elements Elements to include in new bag.
   * @param <T> Type of elements in new bag.
   *
   * @return New RBTBag with provided comparator and elements.
   */
  @SafeVarargs
  public static <T> RBTBag<T> of(Comparator<T> comparator, T... elements) {
    RBTBag<T> rbtBag = new RBTBag<>(comparator);
    rbtBag.insert(elements);
    return rbtBag;
  }

  /**
   * Creates a new RBTBag with natural order and provided elements.
   * <p> Time complexity:  O(n²)
   *
   * @param elements Elements to include in new bag.
   * @param <T> Type of elements in new bag.
   *
   * @return a new RBTBag with natural order and provided elements
   */
  @SafeVarargs
  public static <T extends Comparable<? super T>> RBTBag<T> of(T... elements) {
    return RBTBag.of(Comparator.naturalOrder(), elements);
  }

  /**
   * Creates a new RBTBag with provided comparator and elements in provided iterable.
   * <p> Time complexity: O(n²)
   *
   * @param comparator Comparator defining order of elements in new sorted bag.
   * @param iterable iterable of elements to include in new bag.
   * @param <T> Type of elements in new bag.
   *
   * @return New RBTBag with provided comparator and elements in provided iterable.
   */
  public static <T> RBTBag<T> from(Comparator<T> comparator, Iterable<T> iterable) {
    RBTBag<T> rbtBag = new RBTBag<>(comparator);
    for (T element : iterable) {
      rbtBag.insert(element);
    }
    return rbtBag;
  }

  /**
   * Creates a new RBTBag with natural order and elements in provided iterable.
   * <p> Time complexity:  O(n²)
   *
   * @param iterable iterable of elements to include in new bag.
   * @param <T> Type of elements in new bag.
   *
   * @return a new RBTBag with natural order and pelements in provided iterable.
   */
  public static <T extends Comparable<? super T>> RBTBag<T> from(Iterable<T> iterable) {
    return RBTBag.from(Comparator.naturalOrder(), iterable);
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
    return rbTree.isEmpty();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(n)
   */
  @Override
  public int size() {
    int size = 0;
    for (Pair<T> pair : rbTree.inOrder()) {
      size += pair.occurrences();
    }
    return size;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void delete(T element) {
    rbTree.deleteOrUpdateOrInsert(Pair.withElement(element), pair -> pair.occurrences == 1,
        pair -> Pair.of(element, pair.occurrences - 1), false);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public void clear() {
    rbTree.clear();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public T minimum() {
    if (isEmpty()) {
      throw new NoSuchElementException();
    }
    return rbTree.minimum().element;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public T maximum() {
    if (isEmpty()) {
      throw new NoSuchElementException();
    }
    return rbTree.maximum().element;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void insert(T element) {
    rbTree.deleteOrUpdateOrInsert(Pair.of(element, 1), _ -> false, pair -> Pair.of(element,
        pair.occurrences + 1), true);
  }

  @Override
  public Iterator<T> iterator() {
    return new BagIterator();
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public int occurrences(T element) {
    Pair<T> pair = rbTree.search(Pair.withElement(element));
    return pair == null ? 0 : pair.occurrences();
  }

  private final class BagIterator implements Iterator<T> {
    // Invariant: If value is not null, occurrences > 0 and that is the number
    // of copies of that value left to return. Notice that red black tree can contain
    // nodes for which occurrences are 0.
    Iterator<Pair<T>> it;
    T element;
    int occurrences;

    public BagIterator() {
      it = rbTree.inOrder().iterator();
      advance();
    }

    void advance() {
      while (it.hasNext()) {
        Pair<T> pair = it.next();
        occurrences = pair.occurrences();
        if (occurrences > 0) {
          element = pair.element();
          return;
        }
      }
      element = null;
    }

    public boolean hasNext() {
      return element != null;
    }

    public T next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      T next = element;
      occurrences--;
      if (occurrences < 1) {
        advance();
      }
      return next;
    }
  }
}
