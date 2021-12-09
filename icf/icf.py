from collections import defaultdict
from math import sqrt

import numpy as np
from tqdm import tqdm, trange


class ICF:

    def __init__(self, n_users: int, n_items: int, n_factors: int = 32, std_p: float = 1e-1, std_q: float = 1e-1,
                 std_noise: float = 1e-1):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors

        self.std_p = std_p
        self.std_q = std_q
        self.std_noise = std_noise

        # Creating parameters
        zero_means = np.zeros(self.n_factors)
        ini_cov_p = np.eye(self.n_factors) * (self.std_p ** 2)
        ini_cov_q = np.eye(self.n_factors) * (self.std_q ** 2)
        self.user_factors = np.random.multivariate_normal(zero_means, ini_cov_p, self.n_users)
        self.item_factors = np.random.multivariate_normal(zero_means, ini_cov_q, self.n_items)

        self.lambda_p = (self.std_noise ** 2) / (self.std_p ** 2)
        self.lambda_q = (self.std_noise ** 2) / (self.std_q ** 2)

        self.user_lhs = defaultdict(list)  # used to cumulate the new discovered items for the user
        self.item_lhs = defaultdict(list)  # used to cumulate the new discovered users for the item

        self.user_ratings = defaultdict(list)  # used to cumulate the ratings of the user
        self.item_ratings = defaultdict(list)  # used to cumulate the ratings of the item

    def train(self, train_loader, n_epochs: int = 5):
        # Filling the indexes
        for u_idx, i_idx, rating in tqdm(train_loader, desc='Setting...'):
            u_idx = int(u_idx.item())
            i_idx = int(i_idx.item())
            rating = rating.item()
            self.user_lhs[u_idx].append(i_idx)
            self.item_lhs[i_idx].append(u_idx)

            self.user_ratings[u_idx].append(rating)
            self.item_ratings[i_idx].append(rating)

        for epoch in trange(n_epochs, desc='Epochs'):

            for u_idx, i_idx, rating in tqdm(train_loader, desc='Training'):
                # Sampling and updating the user
                u_idx = int(u_idx.item())
                i_idx = int(i_idx.item())
                rating = rating.item()

                u_items = self.user_lhs[u_idx]
                u_ratings = self.user_ratings[u_idx]
                D_u = self.item_factors[np.array(u_items)]
                A_u = D_u.T @ D_u + self.lambda_p * np.eye(self.n_factors)
                A_inv_u = np.linalg.inv(A_u)
                mu_u = (A_inv_u @ D_u.T) @ np.array(u_ratings)
                sig_u = A_inv_u * (self.std_noise ** 2)

                sampled_u = np.random.multivariate_normal(mu_u, sig_u)
                self.user_factors[u_idx] = sampled_u

                i_users = self.item_lhs[i_idx]
                i_ratings = self.item_ratings[i_idx]
                B_i = self.user_factors[np.array(i_users)]
                P_i = B_i.T @ B_i + self.lambda_q * np.eye(self.n_factors)
                P_inv_i = np.linalg.inv(P_i)
                mu_i = (P_inv_i @ B_i.T) @ np.array(i_ratings)
                sig_i = P_inv_i * (self.std_noise ** 2)

                sampled_i = np.random.multivariate_normal(mu_i, sig_i)
                self.item_factors[i_idx] = sampled_i

            sum_squared_errors = 0.
            for u_idx, i_idx, rating in train_loader:
                u_idx = int(u_idx.item())
                i_idx = int(i_idx.item())
                rating = rating.item()
                dot = (self.user_factors[u_idx] * self.item_factors[i_idx]).sum(axis=-1)
                squared_error = (rating - dot) ** 2
                sum_squared_errors += squared_error

            rsme = sqrt(sum_squared_errors / len(train_loader))
            print(f'RSME is {rsme}')
