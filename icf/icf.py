from math import sqrt

import numpy as np
from tqdm import tqdm, trange


class ICF:

    def __init__(self, n_users: int, n_items: int, n_factors: int = 32, std_p: float = 1e-2, std_q: float = 1e-2,
                 std_noise: float = 1e-3):
        super().__init__()

        # self.mab_dataset = mab_dataset
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

        self.item_mus = np.zeros((self.n_items, self.n_factors))
        self.item_sigs = np.zeros((self.n_items, self.n_factors, self.n_factors))

    def pretrain(self, train_dataset, n_epochs: int = 5):

        # Variables to speed up training

        user_to_lh = dict()
        item_to_lh = dict()
        user_to_rating = dict()
        item_to_rating = dict()

        for u_idx in trange(self.n_users, desc='Setting up user data...'):
            u_lh = train_dataset.lhs[train_dataset.lhs.user_id == u_idx]
            u_items = np.array(u_lh.item_id)
            u_ratings = np.array(u_lh.rating)

            user_to_lh[u_idx] = u_items
            user_to_rating[u_idx] = u_ratings

        for i_idx in trange(self.n_items, desc='Setting up item data...'):
            i_lh = train_dataset.lhs[train_dataset.lhs.item_id == i_idx]
            i_users = np.array(i_lh.user_id)
            i_ratings = np.array(i_lh.rating)

            item_to_lh[i_idx] = i_users
            item_to_rating[i_idx] = i_ratings

        for epoch in trange(n_epochs, desc='Epochs'):

            for u_idx, i_idx, rating in tqdm(train_dataset, desc='Training'):
                u_idx = int(u_idx)
                i_idx = int(i_idx)

                # User step
                u_items = user_to_lh[u_idx]
                u_ratings = user_to_rating[u_idx]
                D_u = self.item_factors[u_items]
                A_u = D_u.T @ D_u + self.lambda_p * np.eye(self.n_factors)
                A_inv_u = np.linalg.inv(A_u)
                mu_u = (A_inv_u @ D_u.T) @ u_ratings
                sig_u = A_inv_u * (self.std_noise ** 2)

                sampled_u = np.random.multivariate_normal(mu_u, sig_u)
                self.user_factors[u_idx] = sampled_u

                i_users = item_to_lh[i_idx]
                i_ratings = item_to_rating[i_idx]

                B_i = self.user_factors[i_users]
                P_i = B_i.T @ B_i + self.lambda_q * np.eye(self.n_factors)
                P_inv_i = np.linalg.inv(P_i)
                mu_i = (P_inv_i @ B_i.T) @ i_ratings
                sig_i = P_inv_i * (self.std_noise ** 2)

                self.item_mus[i_idx] = mu_i
                self.item_sigs[i_idx] = sig_i

                sampled_i = np.random.multivariate_normal(mu_i, sig_i)
                self.item_factors[i_idx] = sampled_i

            print('\n\n\n')
            print(f'End of epoch {epoch}')
            sum_squared_errors = 0.
            for u_idx, i_idx, rating in tqdm(train_dataset, desc='Computing Error..'):
                u_idx = int(u_idx)
                i_idx = int(i_idx)

                dot = (self.user_factors[u_idx] * self.item_factors[i_idx]).sum(axis=-1)
                squared_error = (rating - dot) ** 2
                sum_squared_errors += squared_error

            rsme = sqrt(sum_squared_errors / len(train_dataset))
            print(f'RSME is {rsme}')

            np.savez('./item_distr_parameter.npz', self.item_mus, self.item_sigs)

    def load_params(self):
        dict_arr = np.load('./item_distr_parameter.npz')
        self.item_mus = dict_arr['arr_0']
        self.item_sigs = dict_arr['arr_1']
        print('Loaded')

    def interact_thompson(self, test_dataset):
        # Initializing parameters

        user_b = np.zeros((self.n_users, self.n_factors))  # used only for testing
        user_A = np.repeat((self.lambda_p * np.eye(self.n_factors))[None, :, :], self.n_users,
                           axis=0)  # used only for testing

        cumulative_hit = 0

        for u_idx, i_idx, rating in tqdm(test_dataset, desc='Testing'):
            u_idx = int(u_idx)
            i_idx = int(i_idx)

            # User sampling
            A_inv_u = np.linalg.inv(user_A[u_idx])
            mu_u = A_inv_u @ user_b[u_idx]
            sig_u = A_inv_u * (self.std_noise ** 2)

            sampled_u = np.random.multivariate_normal(mu_u, sig_u)

            # Item sampling/Choice
            max_dot = -np.inf
            i_idx_chose = None
            sampled_i_chose = None
            for i_idx_iter in trange(self.n_items, desc='Choosing the item'):
                mu_i = self.item_mus[i_idx_iter]
                sig_i = self.item_sigs[i_idx_iter]

                sampled_i = np.random.multivariate_normal(mu_i, sig_i)

                dot = (sampled_u * sampled_i).sum()
                if max_dot < dot:
                    max_dot = dot
                    i_idx_chose = i_idx_iter
                    sampled_i_chose = sampled_i

            if i_idx_chose is None or sampled_i_chose is None:
                raise ValueError('Something went wrong during the item search')

            if i_idx_chose == i_idx:
                print('Hit!')
                cumulative_hit += 1

            # else:
            #    print(f'Miss! {i_idx_chose} instead of {i_idx}')

            # Update the parameters accordingly
            user_A[u_idx] = user_A[u_idx] + np.outer(sampled_i_chose, sampled_i_chose)
            user_b[u_idx] = user_b[u_idx] + rating * sampled_i_chose

        print('\n\n')
        print('End of Testing')
        print(f'Number of hits is {cumulative_hit} on {len(test_dataset)}')

    def interact_linucb(self, test_dataset, alpha=1e-1):
        # Initializing parameters

        user_b = np.zeros((self.n_users, self.n_factors))  # used only for testing
        user_A = np.repeat((self.lambda_p * np.eye(self.n_factors))[None, :, :], self.n_users,
                           axis=0)  # used only for testing

        cumulative_hit = 0

        for u_idx, i_idx, rating in tqdm(test_dataset, desc='Testing'):
            u_idx = int(u_idx)
            i_idx = int(i_idx)

            # User pass
            A_inv_u = np.linalg.inv(user_A[u_idx])
            mu_u = A_inv_u @ user_b[u_idx]
            sig_u = A_inv_u * (self.std_noise ** 2)

            # Item Choice

            dots = self.item_mus @ mu_u

            uncer = alpha * np.diag(self.item_mus @ sig_u @ self.item_mus.T)

            i_val = dots + uncer

            i_idx_chose = np.argmax(i_val)

            if i_idx_chose == i_idx:
                print('Hit!')
                cumulative_hit += 1

            # else:
            #   print(f'Miss! {i_idx_chose} instead of {i_idx}')

            # Update the parameters accordingly
            user_A[u_idx] = user_A[u_idx] + np.outer(self.item_mus[i_idx_chose], self.item_mus[i_idx_chose])
            user_b[u_idx] = user_b[u_idx] + rating * self.item_mus[i_idx_chose]

        print('\n\n')
        print('End of Testing')
        print(f'Number of hits is {cumulative_hit} on {len(test_dataset)}')
