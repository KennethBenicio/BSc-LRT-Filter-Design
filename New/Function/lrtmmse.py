import numpy as np
import tensoralgebraezequias as tensoralg

def lrt_mmse(mt_x, train_seq, ndims, order, rank, eps, num_iter):

    # Auxiliary function for linear filtering(Eq.11 of the thesis)
    def mmse_filter(mt_list, rank, order):
        w = 0
        for r in range(rank):
            columns = [mt_list[d][:, [r]] for d in range(order)]
            w += tensoralg.kron(*columns[::-1])
        return w

    n, samples = mt_x.shape
    modes = np.arange(order)  # j modes list
    mt_w = [None] * order  # matrices w_{d}

    # initializing w_{d,r}:
    for d in range(order):  # w_{d} = [w_{d,r} ... ]
        mt_w[d] = np.zeros((ndims[d], rank), dtype=complex)

        # initializing w_{d, r} = [1 0 0 ...]
        mt_w[d][0, :] = np.random.rand(rank)

    # Construction of w_0: vec(W)
    w_aux = mmse_filter(mt_w, rank, order)

    # Storing errors:
    errors = np.zeros(num_iter)

    # Reshaping X as the Tensor:
    dim = order + 1
    shape = ndims + [samples]
    ten_x = np.reshape(mt_x, shape, order='F').T.swapaxes(dim - 2, dim - 1)

    # Filtering algorithm:
    for i in range(num_iter):
        for d in range(order):
            mt_u_dr = [None] * rank
            for r in range(rank):
                # select j modes != d and w_{j,r} columns of w_{d}
                mask = np.ones(order, dtype=bool)
                mask[d] = False
                # hermitian w_{j,r}
                w_jr = [mt_w[j][:, [r]].conj().T
                        for j in range(order) if mask[j]]
                # Build U_{d, r}
                u_dr = tensoralg.m_mode_prod(ten_x, w_jr, modes[mask]).reshape(samples, ndims[d]).T

                # transpose U_{d, r}
                mt_u_dr[r] = u_dr.T

            # Forming U_{d}:
            mt_u_d = np.hstack(*[mt_u_dr]).T
            # Covariances:
            mt_cov = (1 / samples) * mt_u_d @ mt_u_d.conj().T
            vt_cov = (1 / samples) * mt_u_d @ train_seq.conj()

            # Update filter stacked R w_{d, r} columns as RN_{d} x 1:
            w_d_mmse = np.linalg.inv(mt_cov) @ vt_cov

            # Update w_{d}:
            mt_w[d] = tensoralg.unvec(w_d_mmse, ndims[d], rank)

        # Constructing w_i:
        vt_w = mmse_filter(mt_w, rank, order)

        # Error and convergence:
        errors[i] = np.linalg.norm(vt_w - w_aux) ** 2
        if errors[i] <= eps:
            break
        else:
            w_aux = vt_w

    return vt_w, mt_w, ten_x, errors, i