#Optimization

## What is this?

This project involves collection of various optimization algorithms. 
Each Algorithms can solve **arbitrary optimization problem** with specified format.

## Optimization Algorithms

### FPI

FPI or Fixed Point Iteration is basic optimization method solving

$$ \minimize_{x \in \mathbb{R}^d} } f(x)$$

($f$ is differentiable convex function)
This problem could be solved by following Iteration.

$$ x^{k+1} = x^k - \alpha \nabla f(x^k) $$

### PGM and DRS

PGM or Proximal gradient method and DRS or Davis-Yin Splitting method could solve following problem.

$$ \minimize_{x \in \mathbb{R}^d } f(x) + g(x) $$

(For PGM method, $f$ is differentiable convex, $g$ is convex. For DRS method, $f, g$ is convex)
Thus, PGM method could solve following variation.

$$ \minimize_{ \{ x \in \mathbb{R}^d | g(x) \leq 0 \} } f(x) $$

since this problem is equivalent to solving

$$ \minimize_{x \in \mathbb{R}^d } f(x) + \delta_{\{ x \in \mathbb{R}^d | g(x) \leq 0 \} } (x)$$

($\delta_{C} (x) = 0$ for $x \in C$, $\delta_{C} (x) = \infty$ for $x \notin C$
PGM algorithm is

$$x^{k+1} = Prox_{\beta g} (x^k - \beta \nabla f(x^k))$$

DRS algorithm is

$$x^{k+1/2} = Prox_{\beta g} (z^k)$$
$$x^{k+1} = Prox_{\beta f} (2x^{k+1/2} - z^k)$$
$$z^{k+1} = z^{k} + x^{k+1} - x^{k+1/2}$$

Here, $Prox_{f}(x)$ is called 'proximal'. Its formula is

$$Prox_{f}(x) = \argmin_{y \in \mathbb{R}^d} (f(y) + \frac{1}{2} ||x-y||^2 )$$

For $\delta_{\{ x \in \mathbb{R}^d | g(x) \leq 0 \} } (x)$ function, Proximal operator is exactly projection operator.

$$Prox_{f}(x) = \argmin_{y \in \mathbb{R}^d} (f(y) + \frac{1}{2} ||x-y||^2 ) = \argmin_{\{y \in \mathbb{R}^d | g(x) \leq 0 \} ||x-y||}$$

### DYS

DYS or Davis-Yin Splitting method solves problem

$$ \minimize_{x \in mathbb{R}^d} f(x) + g(x) + h(x) $$

DYS algorithm is

$$x^{k+1/2} = Prox_{\beta g} (z^k)$$
$$x^{k+1} = Prox_{\beta f} (2x^{k+1/2} - z^k - \beta \nable h(x^{k+1/2}))$$
$$z^{k+1} = z^k + x^{k+1} - x^{k+1/2}$$

### ADMM

ADMM or alternating direction method of multipliers solves problem

$$ \minimize_{x \in \mathbb{R}^p , y \in \mathbb{R}^q} f(x) + g(y) $$

<center>subject to $Ax + By = c$</center>

($f, g$ is convex function)
Generally, ADMM algorithm does not gaurantee convergence of $x, y$ to $x^*, y^*$. So I selected Linearized ADMM algorithm.

$$x^{k+1} = Prox_{\beta f} (x^k - \beta A^T (u^k + \alpha (Ax^k + By^k - c)))$$
$$y^{k+1} = Prox_{\gamma g} (y^k - \gamma B^T (u^k + \alpha (Ax^{k+1} + By^k - c)))$$
$$u^{k+1} = u^k + \alpha (Ax^{k+1} + By^{k+1} - c)$$

### PDHG and PAPC

PDHG method and PAPC method solves different kind of problem : saddle point problem

$$ \minimize_{x \in \mathbb{R}^n} \maximize_{y \in \mathbb{R}^m } f(x) + y^T Ax - g(y)$$

PDHG algorithm is

$$x^{k+1} = Prox_{\beta f} (x^k - \beta A^T y^k)$$
$$y^{k+1} = Prox_{\gamma g} (y^k + \gamma A(2x^{k+1} - x^k))$$

this algorithm requires $f, g$ to be convex.

PAPC algorithm is

$$y^{k+1} = Prox_{\gamma g} (y^k + \gamma A(x^k - \beta A^T y^k - \beta \nabla f(x^k)))$$
$$x^{k+1} = x^k - \beta A^T y^{k+1} - \beta \nabla f(x^k)$$

this algorithm requires $f$ to be convex differentiable, $g$ to be convex.

### Condat-Vu and PD3O

Condat-Vu method and PD3O method solves following saddle point problem

$$\minimize_{x\in mathbb{R}^n} \maximize_{y \in \mathbb{R}^m } f(x) + y^T Ax - g(y) + h(x)$$

($f, g$ is convex function, $h$ is differentiable and convex)

Condat-Vu algorithm is

$$x^{k+1} = Prox_{\beta f} (x^k - \beta A^T y^k - \beta \nabla h(x^k))$$
$$y^{k+1} = Prox_{\gamma g} (y^k + \gamma A (2x^{k+1} - x^k ))$$

PD3O algorithm is

$$x^{k+1} = Prox_{\beta f} (x^k - \beta A^T y^k - \beta \nabla h(x^k))$$
$$y^{k+1} = Prox_{\gamma g} (y^k + \gamma A (2x^{k+1} - x^k + \beta \nabla h(x^k) - \beta \nabla h(x^{k+1})))$$

## Experiments

In the Experiments_result directory, you could notice experiments done for testing efficiency of algorithms.

First, I implemented LASSO method with two different method : PGM and DRS. You can check LASSO method applied to wine-quality data. I cut-offed wine quality data to seek for the behavior of each algorithms when data size varies.

Second, I implemented sample function for comparison. You could check sample functions below.

<center>PGM vs DRS</center>

$$ f(x_1, \cdots x_n) = \frac{\log{\sum_{i=1}^{m} exp(\frac{1}{i} + \sum_{j=1}^{n} \frac{ij x_j }{(i+j-1)^2 })}}{\sum_{j=1}^{n} \frac{1}{1+\frac{1}{i} e^{x_i}}}$$
$$ g(x_1, \cdots x_n) = \sum_{j=1}^n x_j^2 - 1 $$

Solving the problem : minimize $f$ subject to $g \le 0$

<center>PDHG vs PAPC</center>

$$ f(x_1, \cdots x_n) = \frac{\log{\sum_{i=1}^{m} exp(\frac{1}{i} + \sum_{j=1}^{n} \frac{ij x_j }{(i+j-1)^2 })}}{\sum_{j=1}^{n} \frac{1}{1+\frac{1}{i} e^{x_i}}}$$
$$ g(x_1, \cdots x_n) = \sum_{j=1}^n x_j^2 - 1 $$

Solving the problem :

$$ \minimize_{x \in \mathbb{R}^n} \maximize_{y \in \mathbb{R}^m } f(x) + y^T Ax - g(y)$$

<center>Condat-Vu vs PD3O</center>

$$ f(x_1, \cdots x_n) = \frac{\log{\sum_{i=1}^{m} exp(\frac{1}{i} + \sum_{j=1}^{n} \frac{ij x_j }{(i+j-1)^2 })}}{\sum_{j=1}^{n} \frac{1}{1+\frac{1}{i} e^{x_i}}}$$
$$ g(x_1, \cdots x_n) = \sum_{j=1}^n |x_j|$$
$$ h(x_1, \cdots x_n) = \sum_{j=1}^n x_j^2 - 1 $$

$$ \minimize_{x \in \mathbb{R}^n} \maximize_{y \in \mathbb{R}^m } f(x) + y^T Ax - g(y) + h(x)$$

I compared iteration numbers and time consumption for the iteration for two methods. You can check results.
