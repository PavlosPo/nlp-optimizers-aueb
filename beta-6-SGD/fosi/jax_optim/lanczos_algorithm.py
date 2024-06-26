import jax.numpy as np
import jax.random as random
import jax
from jax.flatten_util import ravel_pytree
from jax import jvp, grad
from jax import config


def lanczos_alg(order, loss, rng_key, k_largest, l_smallest=0, return_precision='32', b_parallel=False):
    """
    Lanczos algorithm for tridiagonalizing a real symmetric matrix, using full reorthogonalization.
    This function returns a function that performs the Lanczos iterations and can be jitted.
    Args:
        order: an integer corresponding to the number of Lanczos steps to take.
        loss: the loss function used to build the hvp operator (loss function to derive).
        rng_key: JAX PRNG key.
        k_largest: an integer corresponding to the required number of the largest eigenvalues and eigenvectors.
        l_smallest: an integer corresponding to the required number of the smallest eigenvalues and eigenvectors.
        return_precision: the algorithm must run in high precision; however, after extracting the eigenvalues and
            eigenvectors we can cast it back to 32/64 bit according to the precision required in return_precision.
        b_parallel: if set to True it means lanczos_alg_jitted is called from within pmap and the batch
            for the hvp is split to multiple devices; therefore, the gradients should be accumulated.
    Returns:
        lanczos_alg_jitted: a function that performs the Lanczos algorith and can be jitted.
    """

    # The algorithm must run in high precision; however, after extracting the eigenvalues and eigenvectors we can
    # cast it back to 32 bit.
    config.update("jax_enable_x64", True)

    def orthogonalization(vecs, w, tridiag, i):
        # Full reorthogonalization.
        # Note that orthogonalization here includes all vectors in vecs, and not just vectors j s.t. j <= i.
        # Since all vectors j s.t. j > i are zeros (vecs is initialized to zeros), there is no impact on w if iteration
        # continues for j > i.
        # However, using the iteration on all the vectors enables us to use jit over this function.
        # Otherwise, we will have to iterate/slice by i, which is not supported by jit.

        # The operation np.dot(np.dot(vecs, w), vecs) is equivalent to multiply (scale) each vector in its own coeff,
        # where coeffs = np.dot(vecs, w) is an array of coeffs (scalars) with shape (order,), and then sum all the
        # scaled vectors.
        w = w - np.dot(np.dot(vecs, w), vecs)  # single vector with the shape of w
        # repeat the full orthogonalization for stability
        w = w - np.dot(np.dot(vecs, w), vecs)  # single vector with the shape of w

        beta = np.linalg.norm(w)

        tridiag = tridiag.at[i, i + 1].set(beta)
        tridiag = tridiag.at[i + 1, i].set(beta)
        vecs = vecs.at[i + 1].set(w / beta)

        return (tridiag, vecs)

    def partial_hvp(i, args, params, batch, v_flat, batch_slice_size):
        '''
        Compute the hvp on slice of the batch
        '''
        w = args
        loss_fn = lambda x: loss(x, (jax.lax.dynamic_slice(batch[0], (i*batch_slice_size,) + (0,)*(len(batch[0].shape)-1), (batch_slice_size,) + batch[0].shape[1:]),
                                     jax.lax.dynamic_slice(batch[1], (i*batch_slice_size,) + (0,)*(len(batch[0].shape)-1), (batch_slice_size,) + batch[1].shape[1:])))
        if b_parallel:
            # Re-use same axis_name as in the call to `pmap(...lanczos_alg_jitted...)`.
            # Note: only support loss functions with mean aggregation. For sum aggregation use psum instead of pmean.
            grad_fn = lambda x: jax.lax.pmean(grad(loss_fn)(x), axis_name='batch')
        else:
            grad_fn = grad(loss_fn)
        hessian_vp = jvp(grad_fn, [params], [v_flat])[1]
        w += ravel_pytree(hessian_vp)[0]
        return w

    def lanczos_iteration(i, args, params, unravel, batch):
        vecs, tridiag = args

        # Get last two vectors
        v = vecs[i]

        # Assign to w the Hessian vector product Hv. Uses forward-over-reverse mode for computing Hv.
        # We assume here that the default precision is 32 bit.
        v_flat = unravel(v) if return_precision == '64' else unravel(v.astype(np.float32))  # convert v to the param tree structure

        # Without batch splitting, in case the hvp on the entire batch fits in memory
        loss_fn = lambda x: loss(x, batch)
        if b_parallel:
            # Re-use same axis_name as in the call to `pmap(...lanczos_alg_jitted...)`.
            # Note: only support loss functions with mean aggregation. For sum aggregation use psum instead of pmean.
            grad_fn = lambda x: jax.lax.pmean(grad(loss_fn)(x), axis_name='batch')
        else:
            grad_fn = grad(loss_fn)
        hessian_vp = jvp(grad_fn, [params], [v_flat])[1]
        w, _ = ravel_pytree(hessian_vp)

        # Batch splitting for large batch size that do no fit in memory; evaluate the hvp on slices of the batch and
        # average the results
        '''w = np.zeros_like(v)
        batch_slice_size = 2
        partial_hvp_fn = lambda i, args: partial_hvp(i, args, params, batch, v_flat, batch_slice_size)
        w = jax.lax.fori_loop(0, batch[0].shape[0] // batch_slice_size, partial_hvp_fn, w)
        w /= (batch[0].shape[0] // batch_slice_size)'''

        w = w.astype(np.float64)

        # Evaluates alpha and update tridiag diagonal with alpha
        alpha = np.dot(w, v)
        tridiag = tridiag.at[i, i].set(alpha)

        # For every iteration except the last one, perform full orthogonalization on w and normalized it (beta is w's
        # norm). Update tridiag secondary diagonals with beta and update vecs with the normalized orthogonal w.
        tridiag, vecs = jax.lax.cond(i + 1 < order,
                                     lambda: orthogonalization(vecs, w, tridiag, i),
                                     lambda: (tridiag, vecs))

        return (vecs, tridiag)

    @jax.jit
    def lanczos_alg_jitted(params, batch):
        """
        Lanczos algorithm for tridiagonalizing a real symmetric matrix, using full reorthogonalization.
        The first time the function is called it is compiled, which can take ~30 second for 10,000,000 parameters
        and order (m) 100.
        Args:
            params: values of the model/function parameters. The gradient of the loss at this params value is used
                in the hvp operator.
            batch: a batch of samples that determines the actual loss function (each loss_i is determined by a
                batch_i of samples, and we use a specific loss_i).
        Returns:
            k_largest_eigenvals: approximate k largest eigenvalues of the Hessian of loss_i at the point params.
            k_largest_eigenvecs: approximate k eigenvectors corresponding to the largest eigenvalues.
            l_smallest_eigenvals: approximate l smallest eigenvalues of the Hessian of loss_i at the point params.
            l_smallest_eigenvecs: approximate l eigenvectors corresponding to the smallest eigenvalues.
        """

        # Initialization
        params_flatten, unravel = ravel_pytree(params)
        num_params = params_flatten.shape[0]
        tridiag = np.zeros((order, order), dtype=np.float64)
        vecs = np.zeros((order, num_params), dtype=np.float64)
        init_vec = random.normal(rng_key, shape=(num_params,), dtype=np.float64)
        init_vec = init_vec / np.linalg.norm(init_vec)
        vecs = vecs.at[0].set(init_vec)

        lanczos_iter = lambda i, args: lanczos_iteration(i, args, params, unravel, batch)
        # Lanczos iterations.
        # Use fori_loop which makes jax compile lanczos_iter() function only once and accelerates the compilation.
        vecs, tridiag = jax.lax.fori_loop(0, order, lanczos_iter, (vecs, tridiag))

        eigs_tridiag, eigvecs_triag = np.linalg.eigh(tridiag)  # eigs_tridiag are also eigenvalues of  the Hessian

        precision = np.float64 if return_precision == '64' else np.float32
        k_largest_eigenvals = eigs_tridiag[order-k_largest:].astype(precision)
        k_largest_eigenvecs = (eigvecs_triag.T[order-k_largest:] @ vecs).astype(precision)
        l_smallest_eigenvals = eigs_tridiag[:l_smallest].astype(precision)
        l_smallest_eigenvecs = (eigvecs_triag.T[:l_smallest] @ vecs).astype(precision)

        return k_largest_eigenvals, k_largest_eigenvecs, l_smallest_eigenvals, l_smallest_eigenvecs

    return lanczos_alg_jitted
