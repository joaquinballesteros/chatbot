package org.uma.ed.datastructures.searchtree;

import java.util.Comparator;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.function.Function;
import java.util.function.Predicate;
import org.uma.ed.datastructures.either.Either;
import org.uma.ed.datastructures.stack.ArrayStack;
import org.uma.ed.datastructures.stack.Stack;

/**
 * Search tree implemented using a balanced left-leaning red-black tree. Nodes are sorted according to their keys and
 * keys are sorted using the provided comparator or their natural order if no comparator is provided.
 *
 * @param <K> Type of keys.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class LLRBT<K> implements SearchTree<K> {
  private static final class Node<K> {
    K key;
    Node<K> left, right;
    boolean isRed;

    Node(K key, boolean isRed) {
      this.key = key;
      this.isRed = isRed;
      this.left = null;
      this.right = null;
    }
  }

  private static <K> Node<K> redNodeOf(K key) {
    return new Node<>(key, true);
  }

  private static <K> boolean isRed(Node<K> node) {
    return node != null && node.isRed;
  }

  private static <K> Node<K> leftRotate(Node<K> node) {
    Node<K> rt = node.right;
    node.right = rt.left;
    rt.left = node;
    rt.isRed = node.isRed;
    node.isRed = true;
    return rt;
  }

  private static <K> Node<K> rightRotate(Node<K> node) {
    Node<K> lt = node.left;
    node.left = lt.right;
    lt.right = node;
    lt.isRed = node.isRed;
    node.isRed = true;
    return lt;
  }

  private static <K> void flipColors(Node<K> node) {
    node.isRed = !node.isRed;
    node.left.isRed = !node.left.isRed;
    node.right.isRed = !node.right.isRed;
  }

  private static <K> Node<K> moveRedLeft(Node<K> node) {
    flipColors(node);
    if(isRed(node.right.left)) {
      node.right = rightRotate(node.right);
      node = leftRotate(node);
      flipColors(node);
    }
    return node;
  }

  private static <K> Node<K> moveRedRight(Node<K> node) {
    flipColors(node);
    if(isRed(node.left.left)) {
      node = rightRotate(node);
      flipColors(node);
    }
    return node;
  }

  private static <K> Node<K> balance(Node<K> node) {
    if(isRed(node.right) && !isRed(node.left)) {
      node = leftRotate(node);
    }
    if(isRed(node.left) && isRed(node.left.left)) {
      node = rightRotate(node);
    }
    if(isRed(node.left) && isRed(node.right)) {
      flipColors(node);
    }
    return node;
  }

  private Node<K> root;
  private int size;
  private final Comparator<K> comparator;

  private LLRBT(Comparator<K> comparator, Node<K> root, int size) {
    this.root = root;
    this.size = size;
    this.comparator = comparator;
  }
  
  public LLRBT(Comparator<K> comparator) {
    this(comparator, null, 0);
  }

  /**
   * Creates an empty red black tree. Keys are sorted according to their natural order.
   * <p> Time complexity: O(1)
   */
  public static <K extends Comparable<? super K>> LLRBT<K> empty() {
    return new LLRBT<K>(Comparator.naturalOrder());
  }

  /**
   * Creates an empty red black tree. Keys are sorted according to provided comparator.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of keys in this search tree.
   */
  public static <K> LLRBT<K> empty(Comparator<K> comparator) {
    return new LLRBT<>(comparator);
  }

  /**
   * Returns a new red black tree with same elements and same structure as argument.
   * <p> Time complexity: O(n log n)
   *
   * @param that binary search tree to be copied.
   *
   * @return a new RedBlackTree with same elements and structure as {@code that}.
   */
  public static <K> LLRBT<K> copyOf(SearchTree<K> that) {
    if (that instanceof LLRBT<K> bst) {
      // use specialized version for RedBlackTree trees
      return copyOf(bst);
    }
    LLRBT<K> copy = new LLRBT<>(that.comparator());
    for (K key : that.preOrder()) {
      copy.insert(key);
    }
    return copy;
  }

  /**
   * Returns a new red black tree with same elements and same structure as argument.
   * <p> Time complexity: O(n)
   *
   * @param that binary search tree to be copied.
   *
   * @return a new RedBlackTree with same elements and structure as {@code that}.
   */
  public static <K> LLRBT<K> copyOf(LLRBT<K> that) {
    return new LLRBT<>(that.comparator, copyOf(that.root), that.size);
  }

  private static <K> Node<K> copyOf(Node<K> node) {
    if (node == null) {
      return null;
    } else {
      Node<K> copy = new Node<>(node.key, node.isRed);
      copy.left = copyOf(node.left);
      copy.right = copyOf(node.right);
      return copy;
    }
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public Comparator<K> comparator() {
    return comparator;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public boolean isEmpty() {
    return root == null;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public int size() {
    return size;
  }

  private static int height(Node<?> node) {
    return node == null ? 0 : 1 + Math.max(height(node.left), height(node.right));
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public int height() {
    return height(root);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(1)
   */
  @Override
  public void clear() {
    root = null;
    size = 0;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public K search(K key) {
    return search(root, key);
  }

  private K search(Node<K> node, K key) {
    if(node == null) {
      return null;
    } else {
      int cmp = comparator.compare(key, node.key);
      if (cmp < 0)
        return search(node.left, key);
      else if (cmp > 0)
        return search(node.right, key);
      else
        return node.key;
    }
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public boolean contains(K key) {
    return search(key) != null;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void insert(K key) {
    root = insert(root, key);
    root.isRed = false;
  }

  private Node<K> insert(Node<K> node, K key) {
    if(node == null) {
      size++;
      node = redNodeOf(key);
    } else {
      int cmp = comparator.compare(key, node.key);
      if(cmp < 0) {
        node.left = insert(node.left, key);
      } else if(cmp > 0) {
        node.right = insert(node.right, key);
      } else {
        // key already in tree
        node.key = key;
      }
      node = balance(node);
    }
    return node;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void deleteMinimum() {
    if (root == null) {
      throw new EmptySearchTreeException("deleteMinimum on empty tree");
    }
    if (!isRed(root.left) && !isRed(root.right)) {
      root.isRed = true;
    }
    root = deleteMinimum(root);
    if (root != null) {
      root.isRed = false;
    }
  }

  private Node<K> deleteMinimum(Node<K> node) {
    if(node.left == null) {
      return null;
    }

    if(!isRed(node.left) && !isRed(node.left.left)) {
      node = moveRedLeft(node);
    }

    node.left = deleteMinimum(node.left);
    return balance(node);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void deleteMaximum() {
    if (root == null) {
      throw new EmptySearchTreeException("deleteMaximum on empty tree");
    }
    if (!isRed(root.left) && !isRed(root.right)) {
      root.isRed = true;
    }

    root = deleteMaximum(root);
    if(root != null) {
      root.isRed = false;
    }
  }

  private Node<K> deleteMaximum(Node<K> node) {
    if(isRed(node.left)) {
      node = rightRotate(node);
    }

    if(node.right == null) {
      return null;
    }

    if(!isRed(node.right) && !isRed(node.right.left)) {
      node = moveRedRight(node);
    }

    node.right = deleteMaximum(node.right);
    return balance(node);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void delete(K key) {
    if (!contains(key)) {
      return;
    }

    if (!isRed(root.left) && !isRed(root.right)) {
      root.isRed = true;
    }

    root = delete(root, key);
    if (root != null)
      root.isRed = false;
  }

  private Node<K> delete(Node<K> node, K key) {
    if (comparator.compare(key, node.key) < 0) {
      if (!isRed(node.left) && !isRed(node.left.left)) {
        node = moveRedLeft(node);
      }
      node.left = delete(node.left, key);
    } else {
      if (isRed(node.left)) {
        node = rightRotate(node);
      }
      if (comparator.compare(key, node.key) == 0 && node.right == null) {
        size--;
        return null;
      }
      if (!isRed(node.right) && !isRed(node.right.left)) {
        node = moveRedRight(node);
      }
      if (comparator.compare(key, node.key) == 0) {
        Node<K> min = minimum(node.right);
        node.key = min.key;
        node.right = deleteMinimum(node.right);
      } else {
        node.right = delete(node.right, key);
      }
    }
    return balance(node);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public K minimum() {
    if (root == null) {
      throw new EmptySearchTreeException("minimum on empty tree");
    }
    return minimum(root).key;
  }

  private static <K> Node<K> minimum(Node<K> node) {
    while (node.left != null) {
      node = node.left;
    }
    return node;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: from O(log n)
   */
  @Override
  public K maximum() {
    if (root == null) {
      throw new EmptySearchTreeException("maxim on empty tree");
    }
    Node<K> node = root;
    while (node.right != null) {
      node = node.right;
    }
    return node.key;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: from O(log n)
   */
  @Override
  public void deleteOrUpdateOrInsert(K key, Predicate<K> delete, Function<K, K> update, boolean insert) {
    int what = deleteOrUpdateOrInsert(root, key, delete, update, insert);
    switch (what) {
      case 1:
        root = insert(root, key);
        break;
      case 2:
        root = delete(root, key);
        break;
      case 3:
        root = delete(root, key);
        root = insert(root, key);
        break;
    }
  }

  private int deleteOrUpdateOrInsert(Node<K> node, K key, Predicate<K> delete, Function<K, K> update, boolean insert) {
    int what;
    if (node == null) {
      if (insert) {
        what = 1; // insert
      } else {
        what = 0;
      }
    } else {
      int cmp = comparator.compare(key, node.key);
      if (cmp < 0) {
        what = deleteOrUpdateOrInsert(node.left, key, delete, update, insert);
      } else if (cmp > 0) {
        what = deleteOrUpdateOrInsert(node.right, key, delete, update, insert);
      } else {
        if (delete.test(node.key)) {
          what = 2; // delete
        } else {
          K newKey = update.apply(node.key);
          if (comparator.compare(newKey, node.key) == 0) {
            node.key = newKey;
            what = 0;
          } else {
            what = 3; // delete and reinsert
          }
        }
      }
    }
    return what;
  }

  // Almost an iterator on keys in tree
  private abstract class Traversal implements Iterator<K> {
    Stack<Either<Node<K>, Node<K>>> stack = new ArrayStack<>();

    public Traversal() {
      if (root != null) {
        save(root);
      }
    }

    abstract void save(Node<K> node);

    public boolean hasNext() {
      return !stack.isEmpty();
    }

    public K next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }

      Either<Node<K>, Node<K>> either = stack.top();
      stack.pop();

      while (either.isRight()) {
        Node<K> node = either.right();
        save(node);
        either = stack.top();
        stack.pop();
      }
      return either.left().key;
    }
  }

  private final class InOrderIterator extends Traversal {
    void save(Node<K> node) {
      // in reverse order, cause stack is LIFO
      if (node.right != null) {
        stack.push(Either.right(node.right));
      }
      stack.push(Either.left(node));
      if (node.left != null) {
        stack.push(Either.right(node.left));
      }
    }
  }

  private final class PreOrderIterator extends Traversal {
    void save(Node<K> node) {
      // in reverse order, cause stack is LIFO
      if (node.right != null) {
        stack.push(Either.right(node.right));
      }
      if (node.left != null) {
        stack.push(Either.right(node.left));
      }
      stack.push(Either.left(node));
    }
  }

  private final class PostOrderIterator extends Traversal {
    void save(Node<K> node) {
      // in reverse order, cause stack is LIFO
      stack.push(Either.left(node));
      if (node.right != null) {
        stack.push(Either.right(node.right));
      }
      if (node.left != null) {
        stack.push(Either.right(node.left));
      }
    }
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public Iterable<K> inOrder() {
    return InOrderIterator::new;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public Iterable<K> preOrder() {
    return PreOrderIterator::new;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public Iterable<K> postOrder() {
    return PostOrderIterator::new;
  }

  /**
   * Returns representation of this search tree as a String.
   */
  @Override
  public String toString() {
    String className = getClass().getSimpleName();
    StringBuilder sb = new StringBuilder(className).append("(");
    toString(sb, root);
    sb.append(")");

    return sb.toString();
  }

  private static void toString(StringBuilder sb, Node<?> node) {
    if (node == null) {
      sb.append("null");
    } else {
      String className = node.getClass().getSimpleName();
      sb.append(className).append("(");
      toString(sb, node.left);
      sb.append(", ");
      sb.append(node.key);
      sb.append(", ");
      sb.append(node.isRed ? "R" : "B");
      sb.append(", ");
      toString(sb, node.right);
      sb.append(")");
    }
  }
}
