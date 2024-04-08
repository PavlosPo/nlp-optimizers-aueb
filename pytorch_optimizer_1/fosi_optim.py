import torch
from torch.optim import Optimizer
from torchopt import pytree
from torch.autograd import forward_ad

class FOSIOptimizer(Optimizer):
    def __init__(self, params, base_optimizer, momentum_func, loss_fn, batch, accumulator_dtype=None,
                 num_iters_to_approx_eigs=100, approx_k=5, approx_l=0, warmup_w=None, alpha=1.0,
                 learning_rate_clip=3.0, device=None):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        accumulator_dtype = None if accumulator_dtype is None else torch.float32
        warmup_w = warmup_w if warmup_w is not None else num_iters_to_approx_eigs
        ese_fn = self.get_ese_fn(loss_fn, approx_k, batch, approx_l, device=device)

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
        lanczos_alg_gitted = self.lanczos_alg(lanczos_order, loss_fn, k_largest, l_smallest, return_precision=return_precision, device=device)

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


    def _approx_learning_rates_and_eigenvectors(self, params):
        if self.batch is not None:
            return self._ese(self.batch, params)
        else:
            # Implement logic for dynamic batch mode if needed
            pass

    def _approx_newton_direction(self, g1, k_eigenvecs, k_learning_rates):
        # Implementation of _approx_newton_direction function
        pass

    def _orthogonalize_vector_wrt_eigenvectors(self, v, k_eigenvecs):
        # Implementation of _orthogonalize_vector_wrt_eigenvectors function
        pass

    def _get_g1_and_g2(self, g, k_eigenvecs):
        # Implementation of _get_g1_and_g2 function
        pass

    def _hvp_forward_ad(self, params, vec, batch):
        torch.nn.utils.vector_to_parameters(vec, self.vec_tree)

        with forward_ad.dual_level():
            dual_params = pytree.tree_map(lambda a, b: forward_ad.make_dual(a, b), params, self.vec_tree)
            loss_val = self.loss_fn(dual_params, batch)
            gradient = torch.autograd.grad(loss_val, dual_params)
            _, hvp = zip(*[forward_ad.unpack_dual(g) for g in gradient])

        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hessian_vec_prod
    
    def _lanczos_alg(self, params, batch):
        k_largest_eigenvals, k_largest_eigenvecs, l_smallest_eigenvals, l_smallest_eigenvecs = self.lanczos_alg_func(params, batch)
        k_eigenvals = torch.cat((l_smallest_eigenvals, k_largest_eigenvals)).to(self.device)
        k_eigenvecs = torch.cat((l_smallest_eigenvecs, k_largest_eigenvecs), 0).to(self.device)
        print(f"lambda_max: {torch.max(k_eigenvals).item()} lrs: {1.0 / k_eigenvals.data} eigenvals: {k_eigenvals.data}")
        return (k_eigenvals, k_eigenvecs)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # Perform FOSI optimization steps here

        return loss