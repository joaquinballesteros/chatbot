package org.uma.ed.datastructures.stack;

import org.uma.ed.datastructures.list.LinkedList;
import org.uma.ed.datastructures.list.List;

/**
 * This class represents a Stack data structure implemented using a linked list of elements.
 * The top of the stack is represented by the first element in the list.
 *
 * @param <T> Type of elements in stack.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class LinkedListStack<T> extends AbstractStack<T> implements Stack<T> {
  /**
   * List of elements in stack.
   */
  private final List<T> elements;

  /*
   * INVARIANT:
   *  - `elements` contains elements in stack in top to bottom order.
   *  - `size` is number of elements in stack.
   */

  /**
   * Creates an empty LinkedListStack.
   * <p> Time complexity: O(1)
   */
  public LinkedListStack() {
    elements = LinkedList.empty();
  }

  /**
   * Creates an empty LinkedListStack.
   * <p> Time complexity: O(1)
   *
   * @param <T> Type of elements in stack.
   *
   * @return an empty LinkedListStack.
   */
  public static <T> LinkedListStack<T> empty() {
    return new LinkedListStack<>();
  }

  /**
   * Creates a LinkedListStack with given elements.
   * <p> Time complexity: O(n)
   *
   * @param elements elements to be added to stack.
   * @param <T> Type of elements in stack.
   *
   * @return a LinkedListStack with given elements.
   */
  @SafeVarargs
  public static <T> LinkedListStack<T> of(T... elements) {
    LinkedListStack<T> stack = new LinkedListStack<>();
    for (T element : elements) {
      stack.push(element);
    }
    return stack;
  }

  /**
   * Creates a LinkedListStack with elements in given iterable.
   * <p> Time complexity: O(n)
   *
   * @param iterable {@code Iterable} of elements to be added to stack.
   * @param <T> Type of elements in iterable.
   *
   * @return a LinkedListStack with elements in given iterable.
   */
  public static <T> LinkedListStack<T> from(Iterable<T> iterable) {
    LinkedListStack<T> stack = new LinkedListStack<>();
    for (T element : iterable) {
      stack.push(element);
    }
    return stack;
  }

  /**
   * Returns a new LinkedListStack with same elements in same order as argument.
   * <p> Time complexity: O(n)
   *
   * @param that LinkedListStack to be copied.
   *
   * @return a new LinkedListStack with same elements and order as {@code that}.
   */
  public static <T> LinkedListStack<T> copyOf(LinkedListStack<T> that) {
    LinkedListStack<T> copy = new LinkedListStack<>();
    for (T element : that.elements) {
      copy.elements.append(element);
    }
    return copy;
  }

  /**
   * Returns a new LinkedListStack with same elements in same order as argument.
   * <p> Time complexity: O(n)
   *
   * @param that Stack to be copied.
   *
   * @return a new LinkedListStack with same elements and order as {@code that}.
   */
  public static <T> LinkedListStack<T> copyOf(Stack<T> that) {
    if (that instanceof LinkedListStack<T> linkedListStack) {
      // use specialized version for LinkedListStack
      return copyOf(linkedListStack);
    }
    Stack<T> temp = new LinkedListStack<>(); // save elements in `that` to `temp`
    while (!that.isEmpty()) {
      temp.push(that.top());
      that.pop();
    }
    LinkedListStack<T> copy = new LinkedListStack<>();
    while (!temp.isEmpty()) {
      that.push(temp.top()); // restore elements in `that`
      copy.push(temp.top()); // copy elements to `copy`
      temp.pop();
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
  public void push(T element) {
    elements.prepend(element);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   *
   * @throws EmptyStackException {@inheritDoc}
   */
  @Override
  public T top() {
    if (isEmpty()) {
      throw new EmptyStackException("top on empty stack");
    }
    return elements.get(0);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   *
   * @throws EmptyStackException {@inheritDoc}
   */
  @Override
  public void pop() {
    if (isEmpty()) {
      throw new EmptyStackException("pop on empty stack");
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
   * Returns a protected iterable over elements in stack.
   */
  protected Iterable<T> elements() {
    return elements;
  }
}


