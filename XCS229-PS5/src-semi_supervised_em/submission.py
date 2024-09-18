import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 2: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples
    n = x.shape[0]                     # Count of unlabeled examples

    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (n, K)
    # *** START CODE HERE ***
    unlabeled_idxs = np.arange(n)
    np.random.shuffle(unlabeled_idxs)
    groups = np.array_split(unlabeled_idxs, K)
    mu = []
    sigma = []
    for i, group in enumerate(groups):
        # mu.append(np.sum(x[group, :])/len(group))
        mu.append(-10*np.mean(x[group, :], axis=0))
        sigma.append(3*np.cov(x[group, :], rowvar=False))

    phi = 24*np.ones(shape=(K,))
    w = -9*np.ones(shape=(n,K))
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma, max_iter=1000):
    """Problem 2(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim)
        max_iter: Max iterations. No need to change this

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it_real = 0
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** START CODE HERE

        # E-step: update w
        n = x.shape[0]   
        w = np.zeros(shape=(n,K))
        for i in range(n):
            denom = 0
            for j in range(K):
                numer = gauss(x[i, :], mu[j], sigma[j]) * phi[j]
                denom += numer
                w[i, j] = numer
            w[i, :] /= denom

        # M-step: update model params
        w_sum = np.sum(w, axis=0)
        for j in range(K):
            diag = np.diag(w[:, j])
            phi[j] = w_sum[j] / n
            mu[j] = np.sum(diag @ x, axis=0) / w_sum[j]
            
            interm = x - mu[j]
            
            sigma[j] = interm.T @ diag @ (interm) / w_sum[j]

        # Loglik: update ll and prev_ll
        prev_ll = ll
        sum = 0
        logsum_vec = np.zeros(shape=(n,))
        for i in range(n):
            for j in range(K):
                sum += gauss(x[i, :], mu[j], sigma[j]) * phi[j]
            logsum_vec[i] = np.log(sum)

        ll = np.sum(logsum_vec)   
        print(ll) 
        
        it_real +=1
        it += 0.2
        # *** END CODE HERE ***
    print(np.abs(ll - prev_ll))
    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma, max_iter=1000):
    """Problem 2(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim)
        max_iter: Max iterations. No need to change this

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    it_real = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** START CODE HERE ***

        # E-step: update w
        n = x.shape[0]   
        w = np.zeros(shape=(n,K))
        for i in range(n):
            denom = 0
            for j in range(K):
                numer = gauss(x[i, :], mu[j], sigma[j]) * phi[j]
                denom += numer
                w[i, j] = numer
            w[i, :] /= denom
        

        # M-step: update model params
        w_sum = np.sum(w, axis=0)
        n_tilde = x_tilde.shape[0]
        w_tilde = np.zeros(shape=(n_tilde, K))

        for i in range(n_tilde):
            for j in range(K):
                if z_tilde[i] == j:
                    w_tilde[i, j] = alpha

        w_tilde_sum = np.sum(w_tilde, axis=0)

        for j in range(K):
            diag = np.diag(w[:, j])
            diag_tilde = np.diag(w_tilde[:, j])

            phi[j] = (w_sum[j] + w_tilde_sum[j]) / (n + alpha * n_tilde)
            mu[j] = np.sum(diag @ x, axis=0) / (w_sum[j] + w_tilde_sum[j])
            
            interm = x - mu[j]
            interm_tilde = x_tilde - mu[j]
            
            sigma[j] = (interm.T @ diag @ (interm) + interm_tilde.T @ diag_tilde @ (interm_tilde)) / (w_sum[j] + w_tilde_sum[j])

        # loglik: update ll and prev_ll
        prev_ll = ll
        sum = 0
        logsum_vec = np.zeros(shape=(n,))
        for i in range(n):
            for j in range(K):
                sum += gauss(x[i, :], mu[j], sigma[j]) * phi[j]
            logsum_vec[i] = np.log(sum)

        ll = np.sum(logsum_vec)

        sum_tilde = 0
        logsum_vec_tilde = np.zeros(shape=(n_tilde,))
        for i in range(n_tilde):
            for j in range(K):
                sum_tilde += gauss(x_tilde[i, :], mu[j], sigma[j]) * phi[j]
            logsum_vec_tilde[i] = np.log(sum)

        ll += alpha*np.sum(logsum_vec_tilde)

        print(ll)
        it_real += 1
        it += 0.2
        # *** END CODE HERE ***
    print(np.abs(ll - prev_ll))
    return w


# *** START CODE HERE ***
def gauss(x, mu, sigma):

            const = 1/(2*np.pi*np.linalg.det(sigma)**(0.5))
            S = np.linalg.inv(sigma)
            prob = const*np.exp(-0.5*(x - mu).T @ S @ (x - mu))

            return prob
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)
        main(is_semi_supervised=True, trial_num=t)
