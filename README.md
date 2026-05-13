# aa222-global-local-path-planning

## Problem Formulation

We are solving a multi-objective optimization problem wherein we search for a set of inputs, $\mathbf{x}$, to optimize the outputs of three objective functions: the energy cost of the mission, $f_e(\mathbf{x})$, the time to the objective, $f_t(\mathbf{x})$, and the event of a crash $f_\infty(\mathbf{x})$. These objective functions are either continuous or binary according to Equation 1:

$$
\begin{gathered} 
    f_e(\mathbf{x}) \in [0, \infty)\\
    f_t(\mathbf{x}) \in [0, \infty)\\
    f_\infty(\mathbf{x}) \in \{0, 1\}
\end{gathered} \tag{1}
$$

We can group these objective functions under one variable, $\mathbf{f}$, which we can define as $\mathbf{f} = [f_e(\mathbf{x}), f_t(\mathbf{x}), f_\infty(\mathbf{x})]$. We can define the ideal optimal point, $\mathbf{y}^*$, according to Equation 2.

$$
\mathbf{y}^* = \begin{bmatrix}
    f_e(\mathbf{x}) = 0 \\
    f_t(\mathbf{x}) = 0 \\
    f_\infty(\mathbf{x}) = 0
\end{bmatrix} = \mathbf{0} \tag{2}
$$

Since $f_e(\mathbf{x})$ and $f_t(\mathbf{x})$ are continuous, we can apply an equal weighting scheme between them to define the overall optimization problem using a weighted exponential sum. This formulation is shown in Equation 3, where $p$ is a hyperparameter of the weighted exponential sum method such that $p \gg 1$.

$$
\begin{aligned}
    \underset{\mathbf{x}}{\text{minimize}} \quad & \sum_{i=0}^{m=2} \frac{1}{m}(f_i(\mathbf{x}) - y_i^*)^p \\
    \text{subject to} \quad & f_\infty(\mathbf{x}) \leq \mathbf{0} \\
                            & \mathbf{x} \in \mathcal{X}
\end{aligned} \tag{3}
$$