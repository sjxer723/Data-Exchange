import numpy as np
import time
import random
from scipy.optimize import linprog
from scipy.optimize import nnls
from utils import *

M = 10
epsilon = 1e-6
epsilon_cnt = 0

def noise():
    # return random.uniform(epsilon, 2 * epsilon)
    global epsilon_cnt, epsilon

    epsilon_cnt+=1
    return epsilon_cnt*epsilon

def get_M():
    global epsilon_cnt, M, epsilon

    epsilon_cnt += 1
    return M - epsilon_cnt*epsilon

def process_csv_to_matrix(file_path, n):
    B = np.zeros((n, 0), dtype=int)
    C = np.zeros((n, 0), dtype=float)

    # Add the first n columns to make B and C in standard form
    for i in range(n):
        b =  np.zeros((n, 1), dtype=int)
        b[i, 0] = 1
        c = np.zeros((n, 1), dtype=float)
        c[i, 0] = 0
        for j in range(n):
            if j != i:
                c[j, 0] = get_M()
        B = np.hstack((B, b))
        C = np.hstack((C, c))
    
    with open(file_path, mode='r', encoding='utf-8') as file:
        coalitions = []
        for line in file:
            row = line.strip().split(',')
            i, j, k = map(int, row[:3])
            u1, u2, u3 = map(float, row[3:6])
            if u1 <= 0  or u2 <= 0 or u3 <= 0:
                continue
            
            coalitions.append((i, j, k))
            b = np.zeros((n, 1), dtype=int)
            b[i, 0] = 1
            b[j, 0] = 1
            b[k, 0] = 1
            c = np.zeros((n, 1), dtype=float)
            c[i, 0] = u1 + noise()
            c[j, 0] = u2 + noise()
            c[k, 0] = u3 + noise()

            for l in range(n):
                if l != i and l != j and l != k:
                    c[l, 0] = get_M()

            B = np.hstack((B, b))
            C = np.hstack((C, c))
    
    # Simplify the coalition set
    coalitions_to_keep = list(range(C.shape[1]))
    for c1 in range(n, C.shape[1]):
        c1 = c1 - n
        i, j, k = coalitions[c1]
        for c2 in range(C.shape[1]):
            if c1 == c2:
                continue
            if not (B[i, c2] == 1 and B[j, c2] == 1 and B[k, c2] == 1):
                continue
            if C[i, c2] >= C[i, c1] and C[j, c2] >= C[j, c1] and C[k, c2] >= C[k, c1]:
                coalitions_to_keep.remove(c1 + n)
                break
    
    return B, C


def CPA(B, C, n):
    # no possible coalitions
    if B.shape[1] == n:
        return 0
    
    # initialization
    card_basis = [i for i in range(n)]
    # Find the column in C with the largest entry on the first row
    j_0 = max([(C[0, i], i) for i in range(n, B.shape[1])])[1]
    if j_0 is None:
        raise ValueError("Fail to find j_0!")
    orid_basis = [i for i in range(1, n)]
    orid_basis.append(j_0)
    
    def cardinal_pivot_update(card_basis, orid_basis):
        # print("INFO: {} {}".format(card_basis, orid_basis))
        j_t = (set(orid_basis) - set(card_basis)).pop()
        
        ## Using the cardinal pivot rule
        B_b = B[:, card_basis]
        B_j_t = B[:, j_t]
    
        bounds = [(0, None) for _ in range(n)]
        x = None
        result = linprog(np.zeros(n), A_eq=B_b, b_eq=np.ones(n), bounds=bounds, method='highs')
        if result.success:
            x = result.x
        else:
            raise("Fail to find the solution x")
        y = np.linalg.solve(B_b, B_j_t)
        
        j_l_idx = min([(x[j] / y[j], j) for j in range(n) if y[j] > epsilon], default=(None, None))[1]
        if j_l_idx is None:
            raise ValueError("Fail to find j_l!")
        j_l_idx = random.choice([j for j in range(n) if y[j] > epsilon and x[j] / y[j] == x[j_l_idx] / y[j_l_idx]])
        j_l = card_basis[j_l_idx]
        card_basis.remove(j_l)
        card_basis.append(j_t)
        # print("Card: Add {}, remove {}".format(j_t, j_l))

        return card_basis, orid_basis

    def ordinal_pivot_update(card_basis, orid_basis):
        j_l = (set(orid_basis) - set(card_basis)).pop()
        
        i_l = None
        for i in range(n):
            u_i = min([C[i, j] for j in orid_basis])
            if C[i, j_l] == u_i:
                i_l = i
                break
        if i_l is None:
            raise ValueError("Fail to find i_l!")
        
        j_r = min([(C[i_l, j], j) for j in orid_basis if j != j_l])[1]
        i_r = None
        # Find the row minizer i_r for the column j_l
        for i in range(n):
            u_i = min([C[i, j] for j in orid_basis])
            if C[i, j_r] == u_i:
                i_r = i
        if i_r is None:
            raise ValueError("Fail to find i_l!")
        
        K = {i for i in range(C.shape[1]) if i not in orid_basis}
        for k in range(C.shape[1]):
            if k in orid_basis:
                continue
            for i in range(n):
                if i == i_r:
                    continue
                u_i = min([C[i, j] for j in orid_basis if j != j_l])
                if C[i, k] <= u_i:
                    K.remove(k)
                    break
        j_star = max([(C[i_r, k], k) for k in K], default=(None, None))[1]
        if j_star is None:
            raise ValueError("Fail to find j_star!")
        orid_basis.append(j_star)
        orid_basis.remove(j_l)
        
        return card_basis, orid_basis
    
    cnt= 0
    while len(set(card_basis) - set(orid_basis)) !=0:
        # print("=========Basis=========")
        # print(card_basis, orid_basis)
        card_basis, orid_basis = cardinal_pivot_update(card_basis, orid_basis)
        cnt +=1
        # print("=========Basis=========")
        # print(card_basis, orid_basis)
        # print(check_stability())
        if len(set(card_basis) - set(orid_basis)) == 0:
            break
        card_basis, orid_basis = ordinal_pivot_update(card_basis, orid_basis)
        
        cnt+=1

    return cnt


if __name__ == '__main__':
    n = 21
    iteration_num = []
    running_time = []

    for replication_id in range(8):
        file_path = "coalition/{}-{}.csv".format(n, replication_id)        
        B, C = process_csv_to_matrix(file_path, n)
        print("Start replication: ", replication_id)

        start_time = time.time()
        info("Number of coaltions: {}".format(C.shape[1]))
        cnt = CPA(B, C, n)
        ok("Finish replication: {}".format(replication_id))
        end_time = time.time()
        
        iteration_num.append(cnt)
        running_time.append(end_time - start_time)

    print(iteration_num, running_time)
    info("Average metrics")
    print(np.mean(iteration_num))
    print(np.mean(running_time))
    info("Medium metrics")
    print(iteration_num)
    print(np.median(iteration_num))
    print(np.median(running_time))