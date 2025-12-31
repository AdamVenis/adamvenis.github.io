---
title: "Solving Medium Sized Traveling Salesman Problems"
mathjax: true
layout: post
published: false
---
The Traveling Salesman Problem (TSP) is known for being one of the most famous "hard" problems to design an algorithm for. It's typically taught in an computer science curriculum, though usually in a fairly superficial way. I found myself needing a deeper understanding of the problem and related algorithms while trying to compete in this kaggle competition (https://www.kaggle.com/competitions/hashcode-drone-delivery). This post explains some of the things I've learned. The goal of this article is to condense weeks of research and experimentation into a short and practical summary, so it may be useful to the next person looking for this knowledge.


In algorithms class, students are mostly taught about algorithms that perform asymptotically "well" and to avoid poor asymptotic behaviour when possible. This guidance is generally valid, but TSP is "NP-complete", meaning there almost certainly [P vs NP](P vs NP) cannot exist an algorithm for it that performs asymptotically well. Nevertheless, it is still a problem that arises in practice, so algorithm designers must do best they can. The label of "NP-complete" suggests that any algorithm will take time to run proportional to two to the power of the size of the input. This would suggest that TSP should be intractible for problem instances larger than, say, 50. In practice, instances with size over 100,000 [link to korean bars] have been solved, using decades of cumulative effort and innovation. 

Bill Cook has done a good job of chronicalling some of the history here https://www.math.uwaterloo.ca/tsp/us/history.html. He explains how larger and larger problems have been solved over time. Among these is a milestone in 1970. Michael Held and Richard Karp, two prominent computer scientists, were the first to fully solve a widely publicized 57-city problem.

If they can do it with the machines of their time, I certainly should be able to solve it now. In 1970, the fastest computer was the [Cray CDC 7600](https://en.wikipedia.org/wiki/CDC_7600), which had an estimated 10MFLOPS. I currently use an AMD Ryzen 9 7900X CPU which has an estimated 3TFLOPS, a 300,000x improvement over 55 years ago.  In the rest of the article, I'll walk through some practical ideas and approaches to TSP, culminating in code that solves this 57-city problem in 15 seconds.


## Methods
We'll use this 57-city problem as a benchmark - let's call it TSP57. Our first few algorithms won't be able to solve the entire problem in a reasonable amount of time, so to measure their performance we'll solve a reduced TSP with only the first few cities, and then incrementally add one city at a time until the runtime exceeds one minute. We'll write the code in Python for ease of implementation. Admittedly it isn't as fast as the Assembly code that would've been written in the 1970s, but this handicap is still nothing compared to the benefits of modern hardware.


## 0. Baseline
The simplest and most direct approach to solving TSP is simply to consider all possible tours - that is, all possible orderings of the N cities, and return one with minimal length (there may be multiple). For N cities, there are N! tours, so this brute force algorithm runs in $$O(N!)$$. By Stirling's approximation(https://en.wikipedia.org/wiki/Stirling%27s_approximation), $$N!$$ grows like $$O(N^N)$$, which is much slower than even a typical exponential $$O(2^N)$$. Here is a concise implementation in Python. 

[code]

For 57 cities, there are $$57! > 10^{76}$$ possible tours. Even if we could evaluate each tour in a single clock cycle, it would take over $$10^59$$ years for the brute force algorithm to solve TSP57 on my CPU. Unfortunately it is predicted that the sun will engulf the earth in just 7.5 billion years, so we can't quite be that patient. 

[graph]
It looks like we can solve up to about 9 cities with this very short, simple code.

## 1. Caching

Caching is one of the most fundamental tools for speeding up programs. Most programs end up redoing the same computations many times over, so by saving and reusing intermediate values, we can avoid further computation and our program will finish quicker. With TSP, iterating over $$N!$$ tours involves a lot of redundant computation, and caching can bring an asymptotic improvement. We'll use [memoization](https://en.wikipedia.org/wiki/Memoization), a simple form of caching. It works by first formulating the problem in a way where it needs to compute $$y = f(x)$$ for many values of $$x$$, and then saves input-output pairs in a hash table so they can be reused, avoid recomputation. With TSP, we'll memoize a function that takes as input both the "current" node and the set of remaining unvisited nodes, and returns the shortest tour that passes through these remaining nodes before returning to the first visited node (the "root"). The algorithm proceeds by calling this function for all subsets of the original input. Each call iterates over all possible next nodes and recursively evaluates results for smaller subsets. 

[maybe a visual of how caching helps, showing multiple subsets of a small problem with a "relies on" relationship]

Think of a tree, where each node represents the decision to visit a particular city next in the tour. Then a tour is a leaf in this tree at depth $$N$$. Thinking recursively, the minimum cost tour is the leaf where at each internal node, the selected branch is the minimum over resulting tour costs[sic]. Structuring the problem this way allows us to reuse computation: if you know the minimum tour for every subset of cities, you can scan linearly at each step to determine the optimal next city.

Tabulating the runtime, we get $$O(n)$$ from each 'next node' and $$2^n$$ subsets for a total runtime of $$O(n\*2^n)$$. Note that this is smaller than $$N!$$, so this yields an asymptotic improvement over the brute force algorithm. 

[code]
[graph]
[notes]

## 2. Search

Since TSP is an NP-complete problem, we can't hope to asymptotically improve beyond $$O(2^N)$$, but we can still make significant improvements in practice. We've explored avoiding recomputing values, but we can do better. In some cases, we don't need to compute the length of certain tours at all if we can guarantee that they can't be the shortest. For example, if the algorithm identifies that some tour has length 100, then any other tour with length over 100 will certainly not be optimal. Essentially, 100 has been established as an upper bound on the optimal tour. If our algorithm selects cities to add to a tour one by one, and some accumulated partial tour length exceeds 100, we don't need to consider any of the (possibly many) tours that start with that partial tour. This allows us to entirely avoid searching through large sets of tours, and could be a big help. This technique is called Branch and Bound (B&B). Here "Branch" refers to exploring a search tree by adding nodes to a tour one by one, and "Bound" refers to making logical arguments to entirely avoid searching certain sets of tours.

[code]
[graph]
[comment that Upper Bounding is done by comparing to the running global min]

So far this isn't an improvement over caching. While it isn't immediately faster, B&B provides a basis on which to add other improvements, which will lead to something faster.


## 3. Lower Bounds
So far we've used upper bounds in B&B to proactively stop searching. We can also use lower bounds to achieve even more pruning, although the concepts and implementation are more complex. We said that if our current partial tour exceeds the upper bound, we can stop searching. Sometimes we can do better, and stop searching even if the current partial tour hasn't exceeded the upper bound, *if we are certain that any completed tour from this point will exceed the upper bound*, even without computing them directly. This amounts to determining some quantity $$L$$ such that any tour consistent with the selections made so far must have length at least $$L$$. Equivalently, if $$L^* := [the shortest remaining tour]$$ we want to find some $$L$$ such that we can guarantee $$L \leq L^*$$, even though we don't know what $$L^*$$ is. Once we establish this $$L$$, if it happens to be larger than our current best tour length $$U$$, we can immediately stop searching down this branch of the search tree and try something else, because we know that we can't hope to improve on our current best.

It's not obvious, but there are many ways to do this and calculate different possible values of $$L$$. If possible, we prefer a larger $$L$$, since it will be more likely to exceed $$U$$ and prune this branch of the search tree. Such lower bounds are called *tighter* when they are closer to $$L^*$$.

Let's define a crude lower bound, and then iteratively refine it. We're computing a lower bound given a partially established tour $$T$$. Let $$n$$ be the number of remaining edges in a tour, and $$\|e\|$$ be the length of the shortest edge among remaining nodes. Then $$\mathcal{l}(T) + n\|e\|$$ is a lower bound at the given internal search point. We know that each edge will cost at least $$c$$, so $$cn$$ is a lower bound for the total cost of a tour. [Distinguish remaining edges]
[TODO: run that code and observe weak performance]

We can do better than selecting the minimum possible edge $$n$$ times. Finishing the tour amounts to picking $$n$$ edges such that (1) the resulting graph is connected, and (2) each node has degree exactly 2. If we relax either of these constraints, that would constitute another lower bound, because we'd be minimizing over a larger set of possibilities. If we only take the second constraint, that's an assignment problem which we could solve with the Hungarian Algorithm. More like that later. If we only take the first constraint, we have something very similar to a [Minimum Spanning Tree](link) problem. Conveniently, this is solvable in $$O(E)$$ time. MST is a lower bound but it isn't quite the one we described though. MST connects the remaining nodes together, but doesn't connect the entire graph. This is a weaker condition, which is why it's a lower bound. By connecting the entire graph, we can add an edge and tighten the bound. [TODO: show result]. We can actually do a little better still. A tour isn't just connected, it's 2-connected, meaning deleting any edge would still leave the remaining graph connected. With $$n$$ nodes unvisited, this requires $$n+1$$ edges. Using only this constraint of 2-connectedness, the minimal such construction can be found by first calculating the MST, and then finding two minimal edges that connect the remaining nodes to the partial tour. This construction has a name in the literature which we can use to conveniently refer back to it - it's called a "1Tree".[TODO: a bit of intuition for why MST + two minimal connection edges is a minimal 2-connected graph].  [TODO: 1tree code results]

Note that MST can be calculated in several different ways. Two of the most popular are Prim's and Kruskall's algorithm. In my testing, Prim's algorithm was a bit faster, but it's possible that a more careful implementation of either could improve on what I've done.

[code, graph]

[it doesn't help but ...]
[notice an interesting tradeoff: some lower bounds may be more expensive to compute, but provide a tighter bound. it's hard to know when this tradeoff is worthwhile, but in principle tighter bounds are worth more computation. a perfectly tight bound reduces total computation from $$O(2^N)$$ to $$O(N^2)$$ by only computing for each candidate and only expanding the correct candidate at every depth]

All this work and still not better than caching yet. We'll revisit this; a tighter lower bound will provide the biggest improvements in this article.


## 4. Cost Reduction
There's a way to think of TSP as selecting entries directly from a distance matrix without using the definition of a tour. TSP can be thought of as the problem of selecting N entries from a distance matrix - one from each row and one from each column, and minimizing the sum of the selected entries. Actually, this isn't quite correct. This definition allows us to select (1,2)(3,4) from (1,2,3,4) which isn't a tour because it's disconnected. 

[diagram of (12)(34) from the corresponding distance matrix]

To fix this formulation, we can add the constraint that there are no cycles of length less than N. From the first part of this definition, any tour must select exactly one entry from each row and column. Therefore, modifying the matrix by adding K to every entry in exactly one row or column will result in the same minimal tours, with length increased by exactly K.

[show square of input, modified input, output, modified output].

Why is this transformation useful? Let's look back at what we have. Profiling with `python -m cProfile -s cumtime run_experiment.py` we see:

[profile]

This shows that the majority ([X]%) of our runtime is spent calculating MSTs, so improving MST calculation will meaningfully improve overall speed. Prim's algorithm works by iteratively find an expanding edge of minimal cost. In TSP, we know that all edges have nonnegative cost, which means if we find an edge with cost 0, it must be minimal and we can short circuit the search.

Coming back to cost reduction, a potential benefit emerges: the more zeroes there are in the distance matrix so long as all entries are nonnegative, the better. Note that for our purposes the diagonal entries are totally ignored by Prim's algorithm, so we don't care if those become negative.

[note that adjacency matrices of graphs are symmetric. if we modify the i'th row without also modifying the corresponding i'th column, the resulting matrix will not be symmetric. so we are limited to adding any penalty to both the i'th row and i'th column for some i.]

Let's try creating as many zero entries possible: if there is any row or column that has all nonzero entries, subtract the minimum value of that row or column from each entry, and keep doing this until it's no longer possible to do so. There's a (faster)[Hungarian Algorithm] way to do this, but our crude approach is already nearly instantaneous, so we can ignore that and move on.

[code]
[results]
Not bad!

There's an abstract idea here that's worth reflecting on. We've found a way to transform our input such that (1)) the output unchanged, and (2) the problem is easier to solve. Even though this transformation preprocessing adds a step and some complexity to our algorithm, it saves time overall.

## 5. Held-Karp Lower Bound

This idea is the least intuitive, so lock in. Held and Karp themselves found a way to compute a tighter lower bound than the 1-Tree Lower Bound we've established, by refining it. 

We saw with cost reduction that we can add a rank-1 matrix $$v^T w$$ to $$D$$ and the TSP length will change by exactly $$\sum(v) + \sum(w)$$. With 1-Trees, that's not true. 1-Trees don't have the properties of selecting exactly two edges incident to each node, so a different penalty will result in a different minimum 1-Tree. However, for any penalty function, *the adjusted 1-Tree cost is still a lower bound for the original problem*. This is the key insight. It means that we can try different penalty values to get different possible lower bounds, and then pick the maximum value to produce the tightest bound.

Once we figure out which penalties to try, we'll have enough information to implement. Given some initial penalty, the gradient of the adjusted 1-Tree cost with respect to the penalty is [deg(i) - 2], so we can perform gradient descent. 

If the 1-Tree has deg(n) == 2 for all nodes, then it found a tour, which is the highest possible lower bound. Otherwise, there will be some nodes with deg > 2 and some with deg < 2. Given some initial penalties, if we add to all penalties for nodes with degree above two and reduce penalties for nodes with degree less than two, if the adjustment or 'step size' is small enough, the lower bound should increase. 

[add penalty -> calculate penalized min 1tree. the penalized min 1tree is a lower bound for all penalties. if there are nodes with degree > 2 in the penalized 1tree, it means we can increase the penalty for that node. similarly if deg(n) < 2 we can decrease the penalty for that node. locally, this update should increase the lower bound. if everything is deg(n) == 2, that's a tour. so do gradient descent on the penalties.]
[note that the max lower bound determined here is not necessarily a tour. Q: why not?]
[Q: is this thing convex?]

[code]
[results]
Nice!

## 6. Additional Improvements
[6a - use tour returned by HK when you can]


[6b - add upper bound to HK]


[6c - WLOG start with x,0,y where y > x]


[6d - search front_node first]


## Conclusion

Further improvements on this implementation could certainly be made, but our original goal has been reached, so I'll stop here. Improvements could come from:
    - faster MST calculation. Many available MST implementations are optimized for large graphs, where we would prioritize lower latency for smaller graphs.
    - better step size calculation [reference]. More care could be put into deciding how to efficiently ascend to a maximal penalty function. Sample details available.
    - upper bound heuristic - when would this matter? Just as a tighter (larger) lower bound increases pruning, so does a tighter (smaller) upper bound. In practice, heuristics like [Lin-Kernighan] are used to establish this as a preprocessing step. In my testing, this doesn't help much in our case, but it could start to help with larger instances [TODO: how much does seeding the perfect upper bound help on TSP57?]
    - I mentioned in the introduction that instances with thousands of nodes can be solved with state of the art solvers. The central technique for these solvers is 'Cutting Planes', which involves treating TSP as an [Integer Program] with combinatorially many constraints, and successively introducing those constraints via [Column Generation] by solving the [Linear Relaxation]. It's beyond the scope of this article, but I'm sure it would be interesting to learn.

Hope this helps! Until next time...

<!--
TODO: 

2025-11-23
==========
- figure out how to embed code.

2025-09-01
==========
- include timings
- include code snippets
- define 1-tree
- flesh out 'additional improvements'
- more editing and writing

===============
is there some hybrid search + caching?
    no because they are fundamentally different. caching is 'bottom up' and requires covering the entire space, whereas search does not.

removing LB from v4 helps lol, the LB is too ineffective.

search is really poor compared to caching lol oh well
it's really just HK LB that matters.

note:
    - low level optimization is nice but algorithmic optimization is better
    - when plotting the new ones, include the old ones as more faint lines


- start with the punchline.


- rather than planning everything out try writing up what we have
    - branch and bound
    - motivation
    - caching  - what exactly are we caching
    - 1tree is complex, kind of. maybe it can be motivated via branch and bound.
    - heuristics
    - 'structure'
    - maybe cut some things. like explaining all the algorithmic stuff, otherwise it might get massive


WLOG start at 0

should I include profiling / profile data?
    - can refer to it, but don't include it in detail.


-->