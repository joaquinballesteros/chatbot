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
 * Search tree implemented using an A. Andersson balanced tree. <p>
 * A. Andersson. Balanced search trees made simple.
 * In Proc. Workshop on Algorithms and Data Structures, pages 60--71.
 * Springer Verlag, 1993.
 * <p>
 * Nodes are sorted according to their keys and keys
 * are sorted using the provided comparator or their natural order if no comparator is provided.
 *
 * @param <K> Type of keys.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class AAT<K> implements SearchTree<K> {
  private static final class Node<K> {
    K key;
    int level;
    Node<K> left, right;

    Node(K key, int level, Node<K> left, Node<K> right) {
      this.key = key;
      this.level = level;
      this.left = left;
      this.right = right;
    }

    Node(K key) {
      this(key, 1, null, null);
    }
  }

  private static <K> int level(Node<K> node) {
    return node == null ? 0 : node.level;
  }

  private Node<K> root;
  private final Comparator<K> comparator;
  private int size;

  private AAT(Comparator<K> comparator, Node<K> root, int size) {
    this.root = root;
    this.comparator = comparator;
    this.size = size;
  }

  public AAT(Comparator<K> comparator) {
    this(comparator, null, 0);
  }
    /**
   * Creates an empty A. Andersson. tree. Keys are sorted according to provided comparator.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of keys in this search tree.
   */
  public static <K> AAT<K> empty(Comparator<K> comparator) {
    return new AAT<>(comparator);
  }

  /**
   * Returns a new A. Andersson. tree with same elements and same structure as argument.
   * <p> Time complexity: O(n log n)
   *
   * @param that binary search tree to be copied.
   *
   * @return a new AAT with same elements and structure as {@code that}.
   */
  public static <K> AAT<K> copyOf(SearchTree<K> that) {
    if (that instanceof AAT<K> bst) {
      // use specialized version for AATs trees
      return copyOf(bst);
    }
    AAT<K> copy = new AAT<>(that.comparator());
    for (K key : that.preOrder()) {
      copy.insert(key);
    }
    return copy;
  }

  /**
   * Returns a new A. Andersson. tree with same elements and same structure as argument.
   * <p> Time complexity: O(n)
   *
   * @param that binary search tree to be copied.
   *
   * @return a new AAT with same elements and structure as {@code that}.
   */
  public static <K> AAT<K> copyOf(AAT<K> that) {
    return new AAT<>(that.comparator, copyOf(that.root), that.size);
  }

  private static <K> Node<K> copyOf(Node<K> node) {
    if (node == null) {
      return null;
    } else {
      return new Node<>(node.key, node.level, copyOf(node.left), copyOf(node.right));
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

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public int height() {
    return height(root);
  }

  private int height(Node<K> node) {
    return node == null ? 0 : 1 + Math.max(height(node.left), height(node.right));
  }

  /**
   * skew removes left horizontal links by rotating right at the parent.
   * No changes are needed to the levels after a skew because the operation
   * simply turns a left horizontal link into a right horizontal link:
   * <pre>
   *          d,2               b,2
   *         /   \             /   \
   *      b,2     e,1  -->  a,1     d,2
   *     /   \                     /   \
   *  a,1     c,1               c,1     e,1
   * </pre>
   * @param node Node to be skewed.
   * @param <K> Type of keys.
   *
   * @return Skewed node.
   */
  private static <K> Node<K> skew(Node<K> node) {
    if (node == null) {
      return null;
    } else if (node.left == null) {
      return node;
    } else if (level(node.left) == level((node))) {
      Node<K> left = node.left;
      node.left = left.right;
      left.right = node;
      return left;
    } else {
      return node;
    }
  }

  /**
   * Unfortunately, a skew could create two consecutive right horizontal links.
   * A split removes consecutive horizontal links by rotating left and increasing
   * the level of the parent:
   * <pre>
   *      b,2                     d,3
   *     /   \                   /   \
   *  a,1     d,2     -->     b,2     e,2
   *         /   \           /   \
   *      c,1     e,2     a,1     c,1
   * </pre>
   * @param node Node to be split.
   * @param <K> Type of keys.
   *
   * @return Split node.
   */
  private static <K> Node<K> split(Node<K> node) {
    if (node == null) {
      return null;
    } else if (node.right == null || node.right.right == null) {
      return node;
    } else if (level(node.right.right) == level(node)) {
      Node<K> right = node.right;
      node.right = right.left;
      right.left = node;
      right.level++;
      return right;
    } else {
      return node;
    }
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void insert(K key) {
    root = insert(root, key);
  }

  private Node<K> insert(Node<K> node, K key) {
    if (node == null) {
      size++;
      return new Node<>(key);
    } else {
      int cmp = comparator.compare(key, node.key);
      if (cmp < 0) {
        node.left = insert(node.left, key);
      } else if (cmp > 0) {
        node.right = insert(node.right, key);
      } else {
        node.key = key;
        return node;
      }
      return fixInsert(node);
    }
  }

  private static <K> Node<K> fixInsert(Node<K> node) {
    node = skew(node);
    node = split(node);
    return node;
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
    while (node != null) {
      int cmp = comparator.compare(key, node.key);
      if (cmp < 0) {
        node = node.left;
      } else if (cmp > 0) {
        node = node.right;
      } else {
        return node.key;
      }
    }
    return null;
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
  public void delete(K key) {
    root = delete(root, key);
  }

  private Node<K> delete(Node<K> node, K key) {
    if (node == null) {
      return null;
    }
    int cmp = comparator.compare(key, node.key);
    if (cmp < 0) {
      node.left = delete(node.left, key);
    } else if (cmp > 0) {
      node.right = delete(node.right, key);
    } else {
      node = delete(node);
    }
    return fixDelete(node);
  }

  private Node<K> delete(Node<K> node) {
    if (node.left == null && node.right == null) {
      size--;
      return null;
    } else if (node.left == null) {
      Node<K> successor = minimum(node.right);
      node.right = delete(node.right, successor.key);
      node.key = successor.key;
    } else {
      Node<K> predecessor = maximum(node.left);
      node.left = delete(node.left, predecessor.key);
      node.key = predecessor.key;
    }
    return node;
  }

  private static <K> Node<K> fixDelete(Node<K> node) {
    if (node == null) {
      return null;
    }
    decreaseLevel(node);
    node = skew(node);
    node.right = skew(node.right);
    if (node.right != null) {
      node.right.right = skew(node.right.right);
    }
    node = split(node);
    node.right = split(node.right);
    return node;
  }

  private Node<K> minimum(Node<K> node) {
    return node.left == null ? node : minimum(node.left);
  }

  private Node<K> maximum(Node<K> node) {
    return node.right == null ? node : maximum(node.right);
  }

  private static <K> void decreaseLevel(Node<K> node) {
    int level = Math.min(level(node.left), level(node.right)) + 1;
    if (level < level(node)) {
      node.level = level;
      if (level < level(node.right)) {
        node.right.level = level;
      }
    }
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
  public K minimum() {
    if (isEmpty()) {
      throw new IllegalStateException("minimum on empty tree");
    }
    return minimum(root).key;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public K maximum() {
    if (isEmpty()) {
      throw new IllegalStateException("maximum on empty tree");
    }
    return maximum(root).key;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void deleteMinimum() {
    if (isEmpty()) {
      throw new IllegalStateException("deleteMinimum on empty tree");
    }
    root = deleteMinimum(root);
  }

  private Node<K> deleteMinimum(Node<K> node) {
    if (node.left == null) {
      size--;
      return node.right;
    }
    node.left = deleteMinimum(node.left);
    return fixDelete(node);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void deleteMaximum() {
    if (isEmpty()) {
      throw new IllegalStateException("deleteMaximum on empty tree");
    }
    root = deleteMaximum(root);
  }

  private Node<K> deleteMaximum(Node<K> node) {
    if (node.right == null) {
      size--;
      return node.left;
    }
    node.right = deleteMaximum(node.right);
    return fixDelete(node);
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void deleteOrUpdateOrInsert(K key, Predicate<K> delete, Function<K, K> update, boolean insert) {
    DeleteUpdateOrInsert dui = new DeleteUpdateOrInsert();
    root = dui.deleteOrUpdateOrInsert(root, key, delete, update, insert);
    if (dui.reinsert) {
      insert(key);
    }
  }

  private final class DeleteUpdateOrInsert {
    boolean reinsert = false;

    private Node<K> deleteOrUpdateOrInsert(Node<K> node, K key, Predicate<K> delete, Function<K, K> update, boolean insert) {
      if (node == null) {
        if (insert) {
          node = new Node<>(key);
          size++;
        }
      } else {
        int cmp = comparator.compare(key, node.key);
        if (cmp < 0) {
          node.left = deleteOrUpdateOrInsert(node.left, key, delete, update, insert);
        } else if (cmp > 0) {
          node.right = deleteOrUpdateOrInsert(node.right, key, delete, update, insert);
        } else {
          if (delete.test(node.key)) {
            node = delete(node);
          } else {
            K newKey = update.apply(node.key);
            if (comparator.compare(newKey, node.key) == 0) {
              node.key = newKey;
            } else {
              node = delete(node);
              reinsert = true;
            }
          }
        }
      }
      return fixDelete(node);
    }
  }

  // An iterator on keys in tree
  private abstract class Traversal implements Iterator<K> {
    Stack<Either<Node<K>, Node<K>>> stack = ArrayStack.empty();

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
      sb.append(node.level);
      sb.append(", ");
      toString(sb, node.right);
      sb.append(")");
    }
  }
}

