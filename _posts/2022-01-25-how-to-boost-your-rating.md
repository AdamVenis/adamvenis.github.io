---
title: "How To Boost Your Rating"
mathjax: true
---
You've been playing a competitive online game for a while. You notice that rating is relatively stagnant, but you'd like to reach the next level, and then take a break. There's a common pattern of players doing this, for example [lichess](https://lichess.org/stat/rating/distribution/blitz) or [dotabuff](https://www.opendota.com/distributions) Ideally you'd like to do so in a way that doesn't require you to get better at the game, because that seems hard. In this blog post we'll investigate how many games it will take you to hit that next level purely by random chance.

## Random Walks of Ratings

We've already seen in a [previous post](https://adamvenis.github.io/2021/11/03/how-accurate-is-your-rating.html) that in the Elo rating system, ratings will randomly fluctuate around their intended value. On average they'll be about 30 points off, but we can also try to calculate a different question, if I want to get to $$x$$ (say, 100) rating points above my true value, how many games will I expect to have to play to do so?

In stochastic process theory, this is called a [hitting time](https://en.wikipedia.org/wiki/Hitting_time) problem. Borrowing the notation from the previous post, let $$S^*$$ is the Markov Chain that represents the dynamics of how our rating changes over time. Then the general formula for $$\tau(x, y)$$, the expected time to first reach state $$y$$ starting from state $$x$$, is:

$$
    \tau(x, y) =
\begin{cases}
    0 &\ \text{if } x = y \\
    1 + \sum_{z \in S^*} A_{x,z} \tau(z, y) & \text{ otherwise } \\
\end{cases}
$$

In our case, we can use a simpler formula. If we're starting at state $$0$$ and want to reach state $$n$$, we'll need to reach state $$n-1$$ first, before then proceeding to $$n$$. So $$\tau(0, n) = \tau(0, n-1) + \tau(n-1, n)$$. Now, before reaching $$n-1$$ we must reach $$n-2$$, and so on, which gives the formula:

$$\tau(0, n) = \sum_{k=0}^{n-1} \tau(k, k+1)$$

Now let's try to simplify again, by using the fact that the only nonzero transitions out of state $$k$$ are $$k-1$$ and $$k+1$$. Referencing formula (1):

$$
\begin{align*}
\tau(k, k+1) &= 1 + \sum_{z \in S^*} A_{x, z} \tau(z, y) \\
&= 1 + A_{k, k+1} \tau(k+1, k+1) + A_{k, k-1} \tau(k-1, k+1) \\
&= 1 + A_{k, k-1} (\tau(k-1, k) + \tau(k, k+1)) \\
(1 - A_{k, k-1}) \tau(k, k+1) &= 1 + A_{k, k-1} \tau(k-1, k) \\
\tau(k, k+1) &= \frac{1 + A_{k, k-1} \tau(k-1, k)}{(1 - A_{k, k-1})}
\end{align*}
$$

```python
def recursive_hitting_time(n, limit):
    return sum(stepping_time(k, limit) for k in range(n))

def stepping_time(n, limit):
    # time to get from state n to n+1
    if n == limit:
        return 1
    else:
        return (1 + p_loss(n) * stepping_time(n-1, limit)) / p_win(n)

def p_loss(n):
    return 1 - p_win(n)

def p_win(n):
    return 1 / (1 + 10 ** (n / 50))

def monte_carlo_hitting_time(n, num_trials=100):
    total_steps = 0
    for _ in range(num_trials):
        steps = 0
        current_position = 0
        while current_position < n:
            current_position += 1 if random.random() < p_win(current_position) else -1
            steps += 1
        total_steps += steps
    return total_steps / num_trials

def analytic_hitting_time(n, limit):
    return 10**(n * (n + 1) / 100) * sum((1 / p_win(k)) * 10**(-k * (k+1) / 100) for k in range(-limit, n))
```

## Conclusion

In summary, we found that on average, it will take you 200 games to gain 50 rating points by pure randomness. We first calculated this with a crude but direct Monte Carlo simulation, and then calculated it again with a more complicated but acurate analytic derivation. I hope you learned something! Until next time...

[Scratchpad extra code]
def recursive_hitting_time(n, limit):
    return sum(stepping_time(k, limit) for k in range(n))

def stepping_time(n, limit):
    # time to get from state n to n+1
    if n == limit:
        return 1
    else:
        return (1 + p_loss(n) * stepping_time(n-1, limit)) / p_win(n)

def p_loss(n):
    return 1 - p_win(n)

def p_win(n):
    return 1 / (1 + 10 ** (n / 50))

def monte_carlo_hitting_time(n, num_trials=100):
    total_steps = 0
    for _ in range(num_trials):
        steps = 0
        current_position = 0
        while current_position < n:
            current_position += 1 if random.random() < p_win(current_position) else -1
            steps += 1
        total_steps += steps
    return total_steps / num_trials

def analytic_hitting_time(n, limit):
    return 10**(n * (n + 1) / 100) * sum((1 / p_win(k)) * 10**(-k* (k+1) / 100) for k in range(-limit, n))

writing notes:
    - common examples, lichess, dotabuff
    - note that league of legends has a wrinkle that the leagues system isn't purely a function of mmr (source?)
    - note that 100 mmr is X% winrate
[End scratchpad]
