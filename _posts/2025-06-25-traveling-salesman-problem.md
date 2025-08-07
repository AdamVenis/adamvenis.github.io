---
title: "Ten practical ideas for faster TSP"
mathjax: true
layout: post
---
The Traveling Salesman Problem (TSP), is particularly historic, known for being one of the most famous "hard" problems to design an algorithm for. It's typically taught in an algorithms curriculum, though usually in a fairly superficial way. I found myself needing a deeper understanding of the problem and related algorithms while trying to compete in this kaggle competition (https://www.kaggle.com/competitions/hashcode-drone-delivery). This post is about what I've learned. The goal is to condense weeks of research, implementation, and experimentation, into a short and practical summary, so it may be useful to the next person looking for this knowledge.


In algorithms class, students are typically taught about algorithms that perform asymptotically "well" and to avoid poor asymptotics when possible. This advice is valid, but TSP is "NP-complete" which means there almost certainly (P vs NP link here) does not exist an algorithm that can perform asymptotically well on it. Nevertheless, it is still a problem that arises in practice, so programmers must do what they can. The label of "NP-complete" suggests that any algorithm will take time to run proportional to two to the power of the size of the input. This would suggest intuitively that TSP should be intractible for problem instances larger than, say 50. In reality, instances with size up to 100k [link] have been solved, using decades of effort and innovation. 

Bill Cook has done a good job of chronicalling the history here https://www.math.uwaterloo.ca/tsp/us/history.html. It explains how larger and larger problem instances have been solved over time. Among the history is a milestone in 1970. Held and Karp, two prominent computer scientists, were able to fully solve a 57-city tour.

If they can do it with the machines of their time, I certainly should be able to. After all, computers have become [~X] faster in the last 55 years. I'll walk through a few fairly intuitive ideas and approaches to TSP, culminating in a program that solves this 57-city problem that we'll call "TSP-57" in 15 seconds.

for each idea:
- motivate
- explain
- benchmark

## Benchmark
We'll use the 57 cities as a benchmark. Most of our prototype algorithms won't be able to solve the entire problem in a reasonable time, so we'll take the first few cities, solve those, and then incrementally add cities until the runtime exceeds one minute. We'll use Python, which is admittedly not a fast language, but Python in 2025 is still thousands of times faster than C in 1970, so we still have a massive handicap[sic]. I'm running this on my AMD Ryzen 9 7900X CPU, running at 4.7GhZ.

## Baseline
The most direct approach to solve TSP is to simply consider all possible tours - that is, all possible orderings of the N cities, and return the one with the minimum length. For N cities, there are N! tours, so this brute force algorithm runs in O(N!). By Stirling's approximation(https://en.wikipedia.org/wiki/Stirling%27s_approximation), N! grows like O(N^N), which is much slower than even a typical exponential O(2^N). Here is a concise implementation in Python. 

[code]

Now for TSP57, there are 57! possible tours, or ~4 * 10^76. If we could evaluate each tour in a single clock cycle (we cant), it would take ~2.7 * 10^59 years for the brute force algorithm to solve TSP57 on my CPU. Unfortunately it is predicted that the sun will engulf the earth in just 7.5 billion years, so we can't wait that long. Instead, to measure the performance of this and other preliminary algorithms, we'll take the first K cities of TSP57 to create smaller problem instances, and solve those for increasing K until the runtime exceeds one minute.

[graph]
[notes]

## Idea 1: Caching

The principle of caching is one of the most fundamental tools in programming to speed up programs.Most programs end up redoing the same computations many times over, so by saving and reusing intermediate values, programs can avoiding further computation and finish quicker. In the case of TSP, iterating over N! tours ends up recomputing a lot. We can avoid this with caching.


[an example of recomputation: [a,b,c,d,e,f] and [b,a,c,d,e,f] only differ by...]
[assuming a fixed root, R - S - x - min_pi(pi(T)) is only dependent on R and x, not S]
[visual]

think of a tree, where each node represents the decision to visit a particular city next in the tour. Then a tour is a leaf in this tree at depth N. Thinking recursively, the minimum cost tour is the leaf where at each internal node, the selected branch is the minimum over resulting tour costs[sic]. Structuring the problem this way allows us to reuse computation: if you know the minimum tour for every subset of cities, you can scan linearly at each step to determine the optimal next city.

There are 2^N subsets, so this algorithm now runs in O(2^N). See that it does indeed outperform the brute force algorithm.

[code]
[graph]
[notes]

## Idea 2: Search

[note somehow that we can't keep caching because caching requires solving the tails to optimality via a "bottoms up" approach, but to search and proactively prune we need a "top down" approach.]
We've explored avoiding computing the same quantities multiple times. Now we'll try to avoid some other types of redundant computation. Imagine you've identified some tour that may not be best, and it has length 100. Now you're considering tours starting with some initial selections S, where S already has length over 100. There is no point in considering any full tours starting with S, because the lengths will only continue to increase above 100, so they cannot possibly be better than your best so far.

In summary, we can track our best seen tour. Then when we're incrementally adding cities to our candidate tours, if the candidate ever exceeds the best seen length, we can "short circuit" rather than searching exhaustively.

This technique is called Branch and Bound. It isn't provably an asymptotic improvement, but in practice it is very broadly applicable and useful.

[code]
[graph]
[notes]
Progress is not always linear. reminds me of stochastic gradient descent. While it isn't immediately faster, Branch and Bound provides the basis on which to add other improvements.

## Idea 3: Include Lower Bounds
So far we've used the upper bound in B&B to proactively stop searching. There's another feature that will enable even more pruning: lower bounding. We said that if our current partial tour exceeds the upper bound, we can stop. In fact, we can stop even if the current partial tour hasn't exceeded the upper bound, *as long as we are certain that any completed tour from this point will exceed the upper bound*, even without computing them directly.


## Idea 5: Preconditioning
1. reduce costs

## Idea 4: Held-Karp Lower Bound

This idea is the least intuitive insight, so lock in. Held and Karp themselves found a way to compute a tighter lower bound than using the 1-Tree Lower Bound. The idea is similar to preconditioning, where modifying the distances to move city i K-units closer to everything, i.e. reducing the i'th row and column by K, reduces the length of any tour by 2K. 


## Idea 6: A handful of other stuff
1. use tour returned by HK when you can
2. add upper bound to HK
3. WLOG start with x,0,y where y > x
4. search front_node first


## Conclusion
future work:
    - faster MST
    - better step size calculation
    - upper bound heuristic - when would this matter?

<!--
is there some hybrid search + caching?


does removing LB from v4 help?!?
    it does. wtf. why.

search is really poor compared to caching lol oh well
is it really just HK LB that makes the difference?
even the 1tree LB is pretty weak

need a moderate amount of complexity that actually improves over caching
to make the post flow

note:
    - low level optimization is nice but algorithmic optimization is better

- when plotting the new ones, include the old ones as more faint lines


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
-->