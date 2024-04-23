import torch
from torch.optim import Optimizer
from torchopt import pytree
from torch.autograd import forward_ad

from typing import Any, Optional, NamedTuple, Callable

import torch
from torchopt.base import GradientTransformation
from lanczos_algorithm import lanczos_alg

# Helper class for the FOSI optimizer
class ScaleByFosiState(NamedTuple):
    base_opt_state: Any  # Optionally, replace Any with OptState for torchopt version >= 0.6.0
    velocity: torch.tensor
    count: torch.tensor
    k_learning_rates: torch.tensor
    k_eigenvecs: torch.tensor
    scaling_factor: torch.tensor

class FOSIOptimizer(Optimizer):
    def __init__(self, params, base_optimizer, momentum_func, loss_fn, batch, accumulator_dtype = None, num_iters_to_approx_eigs=100, approx_k=5, approx_l=0, warmup_w=None, alpha=1.0, learning_rate_clip=3.0, device=None) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.accumulator_dtype = None if accumulator_dtype is None else torch.float32
        self.warmup_w = warmup_w if warmup_w is not None else num_iters_to_approx_eigs
        self.ese_fn = self.get_ese_fn(loss_fn, approx_k, batch, approx_l, device=device)

        self.base_optimizer = base_optimizer(params)
        self.momentum_func = momentum_func
        self.loss_fn = loss_fn
        self.batch = batch
        self.device = device
        self.num_iters_to_approx_eigs = num_iters_to_approx_eigs
        self.approx_k = approx_k
        self.approx_l = approx_l
        self.alpha = alpha
        self.learning_rate_clip = learning_rate_clip

        # Initialize Lanczos algorithm if batch is provided
        if batch is not None:
            self.lanczos_order = 4 * (approx_k + approx_l)
            self.lanczos_alg = lanczos_alg(self.lanczos_order, loss_fn, approx_k, approx_l,
                                            return_precision='32', device=device)
            print("Returned ESE function. Lanczos order (m) is", self.lanczos_order, ".")

        super().__init__(params, {})

    def get_ese_fn(self, loss_fn, k_largest, batch=None, l_smallest=0, return_precision='32', device=torch.device("cpu")):
        # TODO: the following should be max(4 * (k_largest + l_smallest), 2 * int(log(num_params))), however,
        #  num_params is not available at the time of the construction of the optimizer. Note that log(1e+9) ~= 40,
        #  therefore, for num_params < 1e+9 and k>=10 we have 4 * (k_largest + l_smallest) > 2 * log(num_params).
        #  Hence, 4 * (k_largest + l_smallest) is the maximum in our experiments.
        lanczos_order = 4 * (k_largest + l_smallest)
        lanczos_alg_gitted = lanczos_alg(lanczos_order, loss_fn, k_largest, l_smallest, return_precision=return_precision, device=device)

        # The returned ese_fn can be jitted
        if batch is not None:
            # Use static batch mode: all evaluation of the Hessian are done at the same batch
            ese_fn = lambda params: self._ese(lanczos_alg_gitted, batch, params, device)
        else:
            # Use dynamic batch mode: the batch is sent to Lanczos from within the optimizer with args = (params, batch).
            ese_fn = lambda args: self._ese(lanczos_alg_gitted, args[1], args[0], device)

        print("Returned ESE function. Lanczos order (m) is", lanczos_order, ".")
        return ese_fn

    def _ese(self, batch, params):
        k_largest_eigenvals, k_largest_eigenvecs, l_smallest_eigenvals, l_smallest_eigenvecs = self.lanczos_alg(params, batch)
        k_eigenvals = torch.cat((l_smallest_eigenvals, k_largest_eigenvals)).to(self.device)
        k_eigenvecs = torch.cat((l_smallest_eigenvecs, k_largest_eigenvecs), 0).to(self.device)
        print(f"lambda_max: {torch.max(k_eigenvals).item()} lrs: {1.0 / k_eigenvals.data} eigenvals: {k_eigenvals.data}")
        return (k_eigenvals, k_eigenvecs)
    
    # Other methods from your original implementation
    def _hvp_finite_diff(self, params, vec, batch):
        # Implement the hvp_finite_diff method here
        pass

    def _hvp_backward_ad(self, params, vec, batch):
        # Implement the hvp_backward_ad method here
        pass

    def _orthogonalization(self, vecs, w, tridiag, i):
        # Implement the orthogonalization method here
        pass

    def _lanczos_iteration(self, i, args, params, batch):
        # Implement the lanczos_iteration method here
        pass

    def _lanczos_alg_jitted(self, params, batch):
        # Implement the lanczos_alg_jitted method here
        pass


    def _approx_learning_rates_and_eigenvectors(self, params, state : ScaleByFosiState):
        k_eigenvals, k_eigenvecs = self.ese_fn(params)
        k_learning_rates = torch.abs(1.0 / k_eigenvals)
        # Scaling factor for base_opt_deltas, which is clipped k_learning_rates[approx_l] / k_learning_rates[-1]
        scaling_factor = torch.clip(k_learning_rates[self.approx_l] / k_learning_rates[-1], 1.0, self.learning_rate_clip)
        state = ScaleByFosiState(base_opt_state=state.base_opt_state, velocity=state.velocity, count=state.count,
                                 k_learning_rates=k_learning_rates, k_eigenvecs=k_eigenvecs, scaling_factor=scaling_factor)
        return state

    def _approx_newton_direction(self, g1, k_eigenvecs, k_learning_rates):
        # Compute newton_direction (sum of gradient projections on leading eigenvectors scaled by eigenvalues)
        # and batch_gradients (residual of the gradient)
        newton_direction = torch.matmul(k_learning_rates * torch.matmul(k_eigenvecs, g1), k_eigenvecs)
        return newton_direction

    def _orthogonalize_vector_wrt_eigenvectors(v, k_eigenvecs):
        v = v - torch.matmul(torch.matmul(k_eigenvecs, v), k_eigenvecs)
        return v

    def _get_g1_and_g2(g, k_eigenvecs):
        g1 = torch.matmul(torch.matmul(k_eigenvecs, g), k_eigenvecs)  # g1 is the sum of g's projections on k_eigenvecs
        g2 = g - g1  # g2 is orthogonal to g1 and is the sum of g's projection on the rest of the eigenvectors
        return g1, g2
    
    def init_fn(self, params):
        flatten_params = torch.nn.utils.parameters_to_vector(params)
        num_params = flatten_params.shape[0]
        base_opt_state = self.base_optimizer.init(flatten_params)

        velocity = torch.zeros((num_params,), dtype=torch.int32, device=self.device)
        count = torch.zeros((1,), dtype=torch.int32, device=self.device)
        k_learning_rates = torch.zeros((self.approx_k + self.approx_l,), dtype=torch.float32, device=self.device)
        k_eigenvecs = torch.zeros((self.approx_k + self.approx_l, num_params), dtype=torch.float32, device=self.device)
        scaling_factor = torch.ones((1,), dtype=torch.float32, device=self.device)
        return ScaleByFosiState(base_opt_state=base_opt_state, velocity=velocity, count=count,
                                k_learning_rates=k_learning_rates, k_eigenvecs=k_eigenvecs, scaling_factor=scaling_factor)

    def _hvp_forward_ad(self, params, vec, batch):
        """
        hvp_forward_ad for computing the Hessian-vector product using forward-mode automatic differentiation.
        """
        torch.nn.utils.vector_to_parameters(vec, self.vec_tree)

        with forward_ad.dual_level():
            dual_params = pytree.tree_map(lambda a, b: forward_ad.make_dual(a, b), params, self.vec_tree)
            loss_val = self.loss_fn(dual_params, batch)
            gradient = torch.autograd.grad(loss_val, dual_params)
            _, hvp = zip(*[forward_ad.unpack_dual(g) for g in gradient])

        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hessian_vec_prod
    
    # def _lanczos_alg(self, params, batch):
    #     k_largest_eigenvals, k_largest_eigenvecs, l_smallest_eigenvals, l_smallest_eigenvecs = self.lanczos_alg_func(params, batch)
    #     k_eigenvals = torch.cat((l_smallest_eigenvals, k_largest_eigenvals)).to(self.device)
    #     k_eigenvecs = torch.cat((l_smallest_eigenvecs, k_largest_eigenvecs), 0).to(self.device)
    #     print(f"lambda_max: {torch.max(k_eigenvals).item()} lrs: {1.0 / k_eigenvals.data} eigenvals: {k_eigenvals.data}")
    #     return (k_eigenvals, k_eigenvecs)

    # TODO: Correct the values
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grads = p.grad.data
                state = self.state[p]

                updates, state = self.update_fn(grads, state, params=p)

                p.data.add_(updates)

        return loss
    
    def update_fn(self, grads, state : ScaleByFosiState, params):
        if (state.count + 1 >= self.warmup_w) & ((state.count + 1 - self.warmup_w) % self.num_iters_to_approx_eigs == 0):
            state = self._approx_learning_rates_and_eigenvectors(params, state)

        g = torch.nn.utils.parameters_to_vector(grads)
        flatten_params = torch.nn.utils.parameters_to_vector(params)

        g1, g2 = self._get_g1_and_g2(g, state.k_eigenvecs)

        new_velocity = self.momentum_func(g1, state.velocity)
        # Cast the tree to accumulator_dtype
        new_velocity = new_velocity if self.accumulator_dtype is None else new_velocity.type(self.accumulator_dtype)

        newton_direction = self._approx_newton_direction(new_velocity, state.k_eigenvecs, state.k_learning_rates)

        base_opt_deltas, new_base_opt_state = self.base_optimizer.update(g2, state.base_opt_state, params=flatten_params)

        # Reduce from base_opt_deltas its projection on k_eigenvecs
        base_opt_deltas = self._orthogonalize_vector_wrt_eigenvectors(base_opt_deltas, state.k_eigenvecs)

        # Compute the final updates
        torch.nn.utils.vector_to_parameters(state.scaling_factor * base_opt_deltas - self.alpha * newton_direction, grads)
        state = ScaleByFosiState(
            base_opt_state=new_base_opt_state,
            velocity=new_velocity,
            count=state.count + 1,
            k_learning_rates=state.k_learning_rates,
            k_eigenvecs=state.k_eigenvecs,
            scaling_factor=state.scaling_factor
        )

        return grads, state