---
title: "Faster Bin Packing"
mathjax: true
layout: post
---
The Bin Packing Problem (BPP) is a classical algorithms problem in Computer Science. The problem is, given a collection of variable sized items and uniformly sized bins, to pick a bin for each item to be 'packed' into, where the goal is to minimize the number of bins used. Bin packing is considered a fundamental problem due to its simplicity and that many other problems relate to it. It's applied in real world areas like filling up shipping containers for supply chains, creating file backups in media, and technology mapping in FPGA semiconductor chip design. It is, however, an NP-complete problem, which means that there is no existing algorithm that can solve the problem quickly, and it is likely[link to P<>NP] to be impossible for such an algorithm to exist. This post will some of the best theoretical and practical approaches that have been discovered for fast bin packing.

The classical BPP is formulated as an offline algorithm. This means that the algorithm is presented with a fully known set of items and tries to optimally rearrange them into bins. There is a variant called Online Bin Packing, where the algorithm views a single item at a time and incrementally makes an irrevocable decision to either pick an existing bin for it or open a new bin. Another key distinction is between exact and approximate algorithms. An exact algorithm, for a given input, will always return a (possibly non-unique) optimal solution, but may take a very long time to complete. An approximate algorithm, on the other hand, is not guaranteed to return an optimal solution, but will likely complete much more quickly. In a real-world scenario, depending on the tradeoffs for problem at hand, an approximate or online algorithm may be more appropriate. In this post we will only consider exact algorithms in the offline setting. 

[insert some rationale on how we're going to start from the basics and successively introduce improvements]
[insert a point that we're doing 1D bin packing, so we're representing items and bins by real numbers]

Benchmark:
For all of the algorithms we list we're going to set B=100 and randomly generate weights for N items in the range [1, B-1]. We'll measure the runtime of our algorithms for increasing values of N, until it gets so large that the algorithm takes unreasonable long to complete. A simple evaluation we'll use is: for some algorithm A, what's the largest number N such that on average, A solves problems of size N in under one minute?

### 1. Brute Force ###
Fundamentally, a solution to an instance of the BPP (we can refer to an instance as 'a BPP') is a function mapping N items into used bins, so we could conceivably iterate over all possible functions of that form and identify which one is valid, i.e. does not overfill any bin, and uses the fewest bins. Since there are only N items, there must be at most N bins, so there are N^N such functions. Below is some python code that does what we've described, and a graph showing how the runtime grows with input size.

<details>
    <summary<>Python Code</summary>
    ```python
    import collections
    import itertools

    def pack_brute_force(weights, B):
        best_pack = None
        N = len(weights)
        for bin_indexes in itertools.product(*[range(N)] * N):
            pack = collections.defaultdict(list)
            for item, bin_index in zip(weights, bin_indexes):
                pack[bin_index].append(item)

            if all(sum(b) <= B for b in pack.values()) and (
                best_pack is None or len(pack) < len(best_pack)
            ):
                best_pack = list(pack.values())
        return best_pack
    ```
</details>
[TODO: add graph]

* note: can't use variable name `bin` because it's a keyword in python to convert numbers into binary
* note: refer to items and weights interchangeably in the code

#### 2. Limit the number of bins considered ####

We know, on average, that we're going to use a bit more than N/2 bins for our items. Therefore it's wasteful to consider functions that map to all N bins. If we knew approximately the number of bins we need ahead of time, we could significantly reduce the set of functions we need to look at. The risk is that if we guess too low of a number of bins, no assignment will be valid. In this case, we can use a heuristic algorithm that creates a packing pretty-good-but-not-necessarily-perfect number of bins, and then test all functions into that number of bins to see if any functions leave any bins empty, thereby excluding those bins from the result. Decreasing Best Fit (DBF) is a great practical algorithm to approximately solve BPPs. It simply iterates through the items in decreasing order of size and at each step puts the item into the bin that, by that point, is the fullest bin that has capacity to hold the item.

As an aside, it's an interesting exercise to come up with the smallest BPP where DBF does not come up with the optimal answer. In this case let's say the bin capacity and item sizes must be integers, and define one BPP to be smaller than another if either it has fewer items, or has the same number of items and a smaller bin capacity. If you think you've found a minimal example, DM me on twitter about it!

<details>
    <summary>DBF Python Code</summary>
    ```python
    def pack_DBF(weights, B):  # decreasing best fit
    weights = list(reversed(sorted(weights)))
    bins = []
    bin_weights = []
    for w in weights:
        weight_limit = B - w

        eligible_bins = [
            (i, v) for i, v in enumerate(bin_weights) if v <= weight_limit
        ]
        if eligible_bins:
            max_bin_weight_index = max(eligible_bins, key=lambda x: x[1])[0]
            bins[max_bin_weight_index].append(w)
            bin_weights[max_bin_weight_index] += w
        else:
            bins.append([w])
            bin_weights.append(w)

    return bins
    ```
</details>

Now we can use the result from DBF to improve on our brute force algorithm

<details>
    <summary>Brute Force v2</summary>
    ```python
    def pack_brute_force_v2(weights, B):
        best_pack = None
        N = len(weights)
        H = len(pack_DBF(weights, B))
        for bin_indexes in itertools.product(*[range(H)] * N):
            pack = collections.defaultdict(list)
            for item, bin_index in zip(weights, bin_indexes):
                pack[bin_index].append(item)

            if all(sum(b) <= B for b in pack.values()) and (
                best_pack is None or len(pack) < len(best_pack)
            ):
                best_pack = list(pack.values())
        return best_pack
    ```
</details>

[TODO: include graph that shows improvement and text to summarize the improvement]

### 3. Using Search ###

The brute force approaches are very unprincipled: they 
[TODO: include how they consider the 'same' solution multiple times by symmetry of the bins]
it should also be possible to 'short circuit' a solution that has only been partially identified if overflows a bin. The natural way to take advantage of these opportunities is with a search algorithm. We traverse a tree of nodes, where each node corresponds to a partial assignment of items to bins, and the children of each node correspond to associating one additional item to a bin. Then we can ensure that our traversal only visits nodes that correspond to valid partial assignments. We can also eliminate any symmetrical solutions in the following way. First, assume that at each node we've packed exactly the first k items, and that children correspond to different ways to pack the next item in the input. Then, if the current item is being associated with an unopened (aka empty) bin, without loss of generality assume that it is the next bin. With these optimizations, we can hope to see a significant improvement over our existing solution.

<details>
    <summary>Tree Search</summary>
    ```python
    def pack_incremental(weights, B, bins):
        if len(weights) == 0:
            yield bins
            return

        next_weight = weights[0]
        remaining_weights = weights[1:]

        for i, b in enumerate(bins):
            if sum(b) + next_weight <= B:
                bins[i].append(next_weight)
                for packing in pack_incremental(remaining_weights, B, bins):
                    yield packing
                bins[i].pop()

        bins.append([next_weight])
        for packing in pack_incremental(remaining_weights, B, bins):
            yield packing
        bins.pop()


    def pack_search(weights, B):
        bins = []
        best_packing = []
        for packing in pack_incremental(weights, B, bins):
            if not best_packing or len(packing) < len(best_packing):
                best_packing = [list(b) for b in packing]

        return best_packing
    ```
</details>

[TODO: evaluation and enthusiasm]

#### 4. Pre-sort the input ####

One cheap trick is to sort your items in descending order ahead of time. Larger items are more highly constrained on where they can go, so considering them first increases the likelihood of pruning larger branches of the search tree.

<details>
    <summary>Tree Search v2</summary>
    ```python
    def pack_search_v2(weights, B):
        return pack_search(list(reversed(sorted(weights))), B)
    ```
</details>

[TODO: evaluation and enthusiasm]

#### 5. Prune proactively ####

Branch and Bound is a well-known category of algorithms. 
Branch and bound (BB, B&B, or BnB) is a method for solving optimization problems by breaking them down into smaller sub-problems and using a bounding function to eliminate sub-problems that cannot contain the optimal solution. It is an algorithm design paradigm for discrete and combinatorial optimization problems, as well as mathematical optimization. A branch-and-bound algorithm consists of a systematic enumeration of candidate solutions by means of state space search: the set of candidate solutions is thought of as forming a rooted tree with the full set at the root. The algorithm explores branches of this tree, which represent subsets of the solution set. Before enumerating the candidate solutions of a branch, the branch is checked against upper and lower estimated bounds on the optimal solution, and is discarded if it cannot produce a better solution than the best one found so far by the algorithm.

[TODO: fix]

in our case, stop searching at a node if it has chance of beating the previous best packing.

[code, evaluation and enthusiasm]

### 6. Use a solver ###

a bit cheating but the tools are available. also there's a twist that using a solver isn't even the best we can do. Constraint Satisfaction Problem, ORTools, yadda yadda.

[code, evaluation and enthusiasm]

#### 7. Specialize ####
    7) H max bins

[code, evaluation and enthusiasm]

#### 8. Without loss of generality ####

    8) fix the assignment of items with weight > B/2 

[code, evaluation and enthusiasm]

### 9. Branch and Bound on bins directly ###

- you could also reframe the decision tree to complete bins first. this is beneficial because you can track waste and get lower bounds that allow you to prune large branches of the search space.

[code, evaluation and enthusiasm]

#### 10. WLOG next bin contains the next largest item ####

[explanation, code, evaluation and enthusiasm]

#### 11. Track waste ####
    11) do not exceed maximum total waste. this amounts to tracking a lower bound and only excluding an item if the remaining items exceed the lower bound.

[explanation, code, evaluation and enthusiasm]

#### 12. Using Domination ####
    12) if you're going to exclude an item, the sum of the other items you include must exceed the item you excluded

[explanation, code, evaluation and enthusiasm]

#### 13. Handling duplicate sizes ####

    13) handle duplicate values (is this worth the complexity?)

[explanation, code, evaluation and enthusiasm]

## Summary and Conclusion ##


