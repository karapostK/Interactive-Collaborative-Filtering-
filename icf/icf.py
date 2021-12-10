from math import sqrt

import numpy as np
from tqdm import tqdm, trange


class ICF:

    def __init__(self, mab_dataset, n_factors: int = 32, std_p: float = 1e-1, std_q: float = 1e-1,
                 std_noise: float = 1e-1):
        super().__init__()

        self.mab_dataset = mab_dataset
        self.n_users = mab_dataset.n_users
        self.n_items = mab_dataset.n_items
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

        # Variables to speed up training

        self.user_to_lh = dict()
        self.item_to_lh = dict()
        self.user_to_rating = dict()
        self.item_to_rating = dict()

        for u_idx in trange(self.n_users, desc='Setting up user data...'):
            u_lh = self.mab_dataset.lhs[self.mab_dataset.lhs.user_id == u_idx]
            u_items = np.array(u_lh.item_id)
            u_ratings = np.array(u_lh.rating)

            self.user_to_lh[u_idx] = u_items
            self.user_to_rating[u_idx] = u_ratings

        for i_idx in trange(self.n_items, desc='Setting up item data...'):
            i_lh = self.mab_dataset.lhs[self.mab_dataset.lhs.item_id == i_idx]
            i_users = np.array(i_lh.user_id)
            i_ratings = np.array(i_lh.rating)

            self.item_to_lh[i_idx] = i_users
            self.item_to_rating[i_idx] = i_ratings

    def pretrain(self, n_epochs: int = 5):
        # Filling the indexes
        # for u_idx, i_idx, rating in tqdm(train_loader, desc='Setting...'):
        #    u_idx = int(u_idx.item())
        #   i_idx = int(i_idx.item())
        #   rating = rating.item()
        #  self.user_lhs[u_idx].append(i_idx)
        # self.item_lhs[i_idx].append(u_idx)

        # self.user_ratings[u_idx].append(rating)
        # self.item_ratings[i_idx].append(rating)

        for epoch in trange(n_epochs, desc='Epochs'):

            for u_idx, i_idx, rating in tqdm(self.mab_dataset, desc='Training'):
                u_idx = int(u_idx)
                i_idx = int(i_idx)

                # User step
                u_items = self.user_to_lh[u_idx]
                u_ratings = self.user_to_rating[u_idx]
                D_u = self.item_factors[u_items]
                A_u = D_u.T @ D_u + self.lambda_p * np.eye(self.n_factors)
                A_inv_u = np.linalg.inv(A_u)
                mu_u = (A_inv_u @ D_u.T) @ u_ratings
                sig_u = A_inv_u * (self.std_noise ** 2)

                sampled_u = np.random.multivariate_normal(mu_u, sig_u)
                self.user_factors[u_idx] = sampled_u


                i_users = self.item_to_lh[i_idx]
                i_ratings = self.item_to_rating[i_idx]

                B_i = self.user_factors[i_users]
                P_i = B_i.T @ B_i + self.lambda_q * np.eye(self.n_factors)
                P_inv_i = np.linalg.inv(P_i)
                mu_i = (P_inv_i @ B_i.T) @ i_ratings
                sig_i = P_inv_i * (self.std_noise ** 2)

                sampled_i = np.random.multivariate_normal(mu_i, sig_i)
                self.item_factors[i_idx] = sampled_i

            print('\n\n\n')
            print(f'End of epoch {epoch}')
            sum_squared_errors = 0.
            for u_idx, i_idx, rating in tqdm(self.mab_dataset, desc='Computing Error..'):
                u_idx = int(u_idx)
                i_idx = int(i_idx)

                dot = (self.user_factors[u_idx] * self.item_factors[i_idx]).sum(axis=-1)
                squared_error = (rating - dot) ** 2
                sum_squared_errors += squared_error

            rsme = sqrt(sum_squared_errors / len(self.mab_dataset))
            print(f'RSME is {rsme}')
