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
 * Search tree implemented using a balanced red-black tree as described in Introduction to Algorithms by Cormen,
 * Leiserson and Rivest. Nodes are sorted according to their keys and keys are sorted using the provided comparator or
 * their natural order if no comparator is provided.
 *
 * @param <K> Type of keys.
 *
 * @author Pepe Gallardo, Data Structures, Grado en Informática. UMA.
 */
public class RBT<K> implements SearchTree<K> {

  private static final class Node<K> {
    K key;
    boolean red;
    Node<K> left, right, parent;

    Node(K key, boolean red) {
      this.key = key;
      this.red = red;
      this.left = null;
      this.right = null;
      this.parent = null;
    }
  }

  private static boolean isRed(Node<?> node) {
    return node != null && node.red;
  }

  private static boolean isBlack(Node<?> node) {
    return node == null || !node.red;
  }

  private Node<K> root;
  private int size;
  private final Comparator<K> comparator;

  private RBT(Comparator<K> comparator, Node<K> root, int size) {
    this.root = root;
    this.size = size;
    this.comparator = comparator;
  }

  public RBT(Comparator<K> comparator) {
    this(comparator, null, 0);
  }

  /**
   * Creates an empty red black tree. Keys are sorted according to provided comparator.
   * <p> Time complexity: O(1)
   *
   * @param comparator Comparator defining order of keys in this search tree.
   */
  public static <K> RBT<K> empty(Comparator<K> comparator) {
    return new RBT<>(comparator);
  }

  /**
   * Returns a new red black tree with same elements and same structure as argument.
   * <p> Time complexity: O(n log n)
   *
   * @param that binary search tree to be copied.
   *
   * @return a new RBT with same elements and structure as {@code that}.
   */
  public static <K> RBT<K> copyOf(SearchTree<K> that) {
    if (that instanceof RBT<K> bst) {
      // use specialized version for RBT trees
      return copyOf(bst);
    }
    RBT<K> copy = new RBT<>(that.comparator());
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
   * @return a new RBT with same elements and structure as {@code that}.
   */
  public static <K> RBT<K> copyOf(RBT<K> that) {
    return new RBT<>(that.comparator, copyOf(that.root, null), that.size);
  }

  private static <K> Node<K> copyOf(Node<K> node, Node<K> parent) {
    if (node == null) {
      return null;
    } else {
      Node<K> copy = new Node<>(node.key, node.red);
      copy.parent = parent;
      copy.left = copyOf(node.left, copy);
      copy.right = copyOf(node.right, copy);
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

  private final class Finder {
    Node<K> parent = null, node = root;

    Finder(K key) {
      // Search where to insert new node
      while (node != null) {
        parent = node;
        int cmp = comparator.compare(key, node.key);
        if (cmp < 0) {
          node = node.left;
        } else if (cmp > 0) {
          node = node.right;
        } else {
          break;
        }
      }
    }
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void insert(K key) {
    // Search where to insert new node
    Finder finder = new Finder(key);

    if (finder.node != null) {
      finder.node.key = key; // update existing node
    } else  {
      insertBelow(key, finder.parent);
    }
  }

  private void insertBelow(K key, Node<K> parent) {
    // Create new node
    Node<K> node = new Node<>(key, true);
    size++;

    // Insert new node
    node.parent = parent;
    if (parent == null) {
      root = node;
    } else if (comparator.compare(node.key, parent.key) < 0) {
      parent.left = node;
    } else {
      parent.right = node;
    }

    // Fix tree
    insertFixup(node);
  }

  private void insertFixup(Node<K> node) {
    while (isRed(node.parent)) {
      if (node.parent == node.parent.parent.left) {
        Node<K> uncle = node.parent.parent.right;
        if (isRed(uncle)) {
          node.parent.red = false;
          uncle.red = false;
          node.parent.parent.red = true;
          node = node.parent.parent;
        } else {
          if (node == node.parent.right) {
            node = node.parent;
            leftRotate(node);
          }
          node.parent.red = false;
          node.parent.parent.red = true;
          rightRotate(node.parent.parent);
        }
      } else {
        Node<K> uncle = node.parent.parent.left;
        if (isRed(uncle)) {
          node.parent.red = false;
          uncle.red = false;
          node.parent.parent.red = true;
          node = node.parent.parent;
        } else {
          if (node == node.parent.left) {
            node = node.parent;
            rightRotate(node);
          }
          node.parent.red = false;
          node.parent.parent.red = true;
          leftRotate(node.parent.parent);
        }
      }
    }
    root.red = false;
  }

  private void leftRotate(Node<K> node) {
    Node<K> rt = node.right;
    node.right = rt.left;
    if (rt.left != null) {
      rt.left.parent = node;
    }
    rt.parent = node.parent;
    if (node.parent == null) {
      root = rt;
    } else if (node == node.parent.left) {
      node.parent.left = rt;
    } else {
      node.parent.right = rt;
    }
    rt.left = node;
    node.parent = rt;
  }

  private void rightRotate(Node<K> node) {
    Node<K> lt = node.left;
    node.left = lt.right;
    if (lt.right != null) {
      lt.right.parent = node;
    }
    lt.parent = node.parent;
    if (node.parent == null) {
      root = lt;
    } else if (node == node.parent.right) {
      node.parent.right = lt;
    } else {
      node.parent.left = lt;
    }
    lt.right = node;
    node.parent = lt;
  }

  private Node<K> searchNode(K key) {
    Node<K> node = root;
    while (node != null) {
      int cmp = comparator.compare(key, node.key);
      if (cmp < 0) {
        node = node.left;
      } else if (cmp > 0) {
        node = node.right;
      } else {
        break;
      }
    }
    return node;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public K search(K key) {
    Node<K> node = searchNode(key);
    return node == null ? null : node.key;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public boolean contains(K key) {
    Node<K> node = searchNode(key);
    return node != null;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: O(log n)
   */
  @Override
  public void delete(K key) {
    Node<K> node = searchNode(key);
    if (node != null) {
      deleteNode(node);
    }
  }

  private void transplant(Node<K> destination, Node<K> source) {
    if (destination.parent == null) {
      root = source;
    } else if (destination == destination.parent.left) {
      destination.parent.left = source;
    } else {
      destination.parent.right = source;
    }
    if (source != null) {
      source.parent = destination.parent;
    }
  }

  private void deleteNode(Node<K> node) {
    Node<K> removedOrMoved = node, nodeToFixUp, nodeToFixUpParent;
    boolean removedOrMovedOriginalRed = removedOrMoved.red;

    if (node.left == null) {
      nodeToFixUp = node.right;
      transplant(node, node.right);
      nodeToFixUpParent = node.parent;
    } else if (node.right == null) {
      nodeToFixUp = node.left;
      transplant(node, node.left);
      nodeToFixUpParent = node.parent;
    } else {
      removedOrMoved = minimumNode(node.right);
      removedOrMovedOriginalRed = removedOrMoved.red;
      nodeToFixUp = removedOrMoved.right;
      if (removedOrMoved != node.right) {
        nodeToFixUpParent = removedOrMoved.parent;
        transplant(removedOrMoved, removedOrMoved.right);
        removedOrMoved.right = node.right;
        removedOrMoved.right.parent = removedOrMoved;
      } else {
        nodeToFixUpParent = removedOrMoved;
      }
      transplant(node, removedOrMoved);
      removedOrMoved.left = node.left;
      removedOrMoved.left.parent = removedOrMoved;
      removedOrMoved.red = node.red;
    }

    if (!removedOrMovedOriginalRed) {
      deleteFixup(nodeToFixUp, nodeToFixUpParent);
    }
    size--;
  }

  private void deleteFixup(Node<K> node, Node<K> nodeParent) {
    while (node != root && isBlack(node)) {
      if (node == nodeParent.left) {
        Node<K> sibling = nodeParent.right;
        if (sibling.red) {
          sibling.red = false;
          nodeParent.red = true;
          leftRotate(nodeParent);
          sibling = nodeParent.right;
        }
        if (isBlack(sibling.left) && isBlack(sibling.right)) {
          sibling.red = true;
          node = nodeParent;
        } else {
          if (isBlack(sibling.right)) {
            sibling.left.red = false;
            sibling.red = true;
            rightRotate(sibling);
            sibling = nodeParent.right;
          }
          sibling.red = nodeParent.red;
          nodeParent.red = false;
          sibling.right.red = false;
          leftRotate(nodeParent);
          node = root;
        }
      } else {
        Node<K> sibling = nodeParent.left;
        if (sibling.red) {
          sibling.red = false;
          nodeParent.red = true;
          rightRotate(nodeParent);
          sibling = nodeParent.left;
        }
        if (isBlack(sibling.right) && isBlack(sibling.left)) {
          sibling.red = true;
          node = nodeParent;
        } else {
          if (isBlack(sibling.left)) {
            sibling.right.red = false;
            sibling.red = true;
            leftRotate(sibling);
            sibling = nodeParent.left;
          }
          sibling.red = nodeParent.red;
          nodeParent.red = false;
          sibling.left.red = false;
          rightRotate(nodeParent);
          node = root;
        }
      }
      nodeParent = node.parent;
    }
    if(node != null)
      node.red = false;
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
  public void deleteMinimum() {
    if (root == null) {
      throw new EmptySearchTreeException("deleteMinimum on empty tree");
    }
    Node<K> node = minimumNode(root);
    deleteNode(node);
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
    Node<K> node = maximumNode(root);
    deleteNode(node);
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
    return minimumNode(root).key;
  }

  private static <K> Node<K> minimumNode(Node<K> node) {
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
      throw new EmptySearchTreeException("maximum on empty tree");
    }
    return maximumNode(root).key;
  }

  private static <K> Node<K> maximumNode(Node<K> node) {
    while (node.right != null) {
      node = node.right;
    }
    return node;
  }

  /**
   * {@inheritDoc}
   * <p> Time complexity: from O(log n)
   */
  @Override
  public void deleteOrUpdateOrInsert(K key, Predicate<K> delete, Function<K, K> update, boolean insert) {
    Finder finder = new Finder(key);

    if (finder.node != null) {
      if (delete.test(finder.node.key)) {
        deleteNode(finder.node); // delete
      } else {
        K newKey = update.apply(finder.node.key);
        if (comparator.compare(newKey, finder.node.key) == 0) {
          finder.node.key = newKey; // same order so just update
        } else {
          // different order so delete and reinsert
          deleteNode(finder.node); // delete
          insert(key); // and reinsert
        }
      }
    } else if (insert) {
      insertBelow(key, finder.parent);
    }
  }

  // Almost an iterator on keys in tree
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
      sb.append(node.red ? "R" : "B");
      sb.append(", ");
      toString(sb, node.right);
      sb.append(")");
    }
  }
}

