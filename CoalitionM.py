import re
import random
import math
import csv
import time
import numpy as np
import xml.etree.ElementTree as ET
import networkx as nx
from scipy.optimize import minimize

import networkx as nx


def length2delay(length, midpoint=50, k=0.1):
    return 1 / (1 + math.exp(-k * (length - midpoint)))

def bfs_layers(G, source):
    # Perform a BFS from the source node and return nodes by layers
    layers = {}
    visited = {source}
    queue = [(source, 0)]  # (node, layer)
    
    while queue:
        node, layer = queue.pop(0)
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node)
        
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, layer + 1))
    
    return layers

def bfs_path(G, source, target):
    # Perform BFS to find the path from source to target
    # Returns the path from source to target if exists, otherwise None
    parent = {source: None}
    queue = [source]
    
    while queue:
        node = queue.pop(0)
        
        if node == target:
            # Reconstruct the path from source to target
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()  # Reverse the path to go from source to target
            return path
        
        for neighbor in G.neighbors(node):
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    
    return None  # If no path exists

def sample_path(G):
    def get_samples(path):
        edge_samples = {}

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            delay = length2delay(G[u][v]['weight'])

            num_of_sample = random.randint(4, 9)
            # samples = np.random.normal(loc=delay, scale=1, size=num_of_sample)
            edge_samples[(u, v)] = num_of_sample #samples
    
        return edge_samples
    
    while True:
        u = random.choice(list(G.nodes))

        layers = bfs_layers(G, u) # Get BFS tree

        max_depth = max(layers.keys())
        if max_depth < 5:
            continue
        t = random.randint(5, max_depth)

        if t in layers:
            v = random.choice(layers[t])
            path = bfs_path(G, u, v)
            
            if path:
                return path, get_samples(path)



class CPA():
    def __init__(self, graphml_file, _N):
        original_G = nx.read_graphml(graphml_file)
        self.G = nx.DiGraph()
        for source, target, data in original_G.edges(data=True):
            # if not data.get('length'):
            #     continue
            # length = float(data['length'])
            self.G.add_edge(source, target, weight=random.uniform(0,1))
        self.N = _N    # number of agents
        self.paths = []
        self.delay_samples = []
        for _ in range(_N):
            path, delay_sample = sample_path(self.G)
            self.paths.append(path)
            self.delay_samples.append(delay_sample)
        # self.cost = [0.001 for _ in range(_N)]
        self.cost = [random.uniform(0, 0.05) for _ in range(_N)]
        self.possible_coalitions = dict() # coalition -> (utility, exchange)
        self.possible_exchanges = dict()
        self.invoke_oracle_num = 0
        self.num_of_coalitions = 0
        
    def utility(self, i, j, k, agent, common_edges, x):
        utility = 0
        common_edge12 = common_edges[(i, j)]
        common_edge13 = common_edges[(i, k)]
        common_edge23 = common_edges[(j, k)]
        x12, x13, x21, x23, x31, x32 = x

        edge_set1, edge_set2 = None, None 
        out_share1, out_share2 = 0, 0
        in_share1, in_share2 = 0, 0
        remaining_agents = [i, j, k]
        remaining_agents.remove(agent)
        if agent == i:
            edge_set1, edge_set2 = common_edge12, common_edge13
            out_share1, out_share2 = x12, x13
            in_share1, in_share2 = x21, x31
        elif agent == j:
            edge_set1, edge_set2 = common_edge12, common_edge23
            out_share1, out_share2 = x21, x23
            in_share1, in_share2= x12, x32
        elif agent == k:
            edge_set1, edge_set2 = common_edge13, common_edge23
            out_share1, out_share2 = x31, x32
            in_share1, in_share2 = x13, x23
        else:
            raise ValueError("Invalid agent index.")
        
        for edge in (edge_set1 & edge_set2):
            variance = self.G[edge[0]][edge[1]]['weight']
            n = {t: self.delay_samples[t][(edge[0], edge[1])] for t in [i, j, k]}
            
            utility += variance / n[agent] - variance / (n[agent] + in_share1 * n[remaining_agents[0]] + in_share2 * n[remaining_agents[1]])\
                - self.cost[agent] * (out_share1 + out_share2) * n[agent]
        
        for edge in (edge_set1 - edge_set2):
            variance = self.G[edge[0]][edge[1]]['weight']
            n = {t: self.delay_samples[t][(edge[0], edge[1])] for t in [agent, remaining_agents[0]]}
            
            if n[agent] == 0 or n[agent] + in_share1 * n[remaining_agents[0]] == 0:
                print(f"n[agent]={n[agent]}, in_share1={in_share1}, n[remaining_agents[0]]={n[remaining_agents[0]]}")
            utility += variance / n[agent] - variance / (n[agent] + in_share1 * n[remaining_agents[0]]) - self.cost[agent] * out_share1 * n[agent]

        for edge in (edge_set2 - edge_set1):
            variance = self.G[edge[0]][edge[1]]['weight']
            n = {t: self.delay_samples[t][(edge[0], edge[1])] for t in [agent, remaining_agents[1]]}
            if n[agent] == 0 or n[agent] + in_share1 * n[remaining_agents[1]] == 0:
                print(f"n[agent]={n[agent]}, in_share1={in_share2}, n[remaining_agents[0]]={n[remaining_agents[1]]}")
            utility += variance / n[agent] - variance / (n[agent] + in_share2 * n[remaining_agents[1]]) - self.cost[agent] * out_share2 * n[agent]

        return utility
        


    def determine_feasibility(self, i, j, k, common_edges, u1, u2, u3):
        # For statistics, measure the number of times the oracle is invoked
        self.invoke_oracle_num += 1
        # if self.invoke_oracle_num % 1000 == 0:
        #     print("Invoke oracle {}".format(self.invoke_oracle_num))
        
        def constraint1(vars): # utility constraint of agent i
            return self.utility(i, j, k, i, common_edges, vars) - u1
        
        def constraint2(vars): # utility constraint of agent j
            return self.utility(i, j, k, j, common_edges, vars) - u2

        def constraint3(vars): # utility constraint of agent k
            return self.utility(i, j, k, k, common_edges, vars) - u3

        initial_guess = [1, 1, 1, 1, 1, 1]

        cons = [{'type': 'ineq', 'fun': lambda vars: 1 - np.array(vars)},
                {'type': 'ineq', 'fun': lambda vars: np.array(vars)},
                {'type': 'ineq', 'fun': constraint1},
                {'type': 'ineq', 'fun': constraint2},    
                {'type': 'ineq', 'fun': constraint3}]

        result = minimize(lambda vars: -sum(vars), initial_guess, constraints=cons)

        if result.success:
            x12, x13, x21, x23, x31, x32 = result.x
            return True, (x12, x13, x21, x23, x31, x32)
        else:
            return False, None
        
    def binary_search_and_collect(self, max_value, fixed1, fixed2, common_edges, i, j, k, idx):
        possible_coalitions = set()
        # , possible_exchanges = set(), set()
        low, high = 0, max_value
        is_feasible = False
        utility_vector, sol = None, None
        while high - low > 0.1:
            mid = (low + high) / 2
            if idx == 0:
                u1, u2, u3 = mid, fixed1, fixed2
            elif idx == 1:
                u1, u2, u3 = fixed1, mid, fixed2
            else:
                u1, u2, u3 = fixed1, fixed2, mid

            if any(u1 <= _u1 and u2 <= _u2 and u3 <= _u3 for ((_u1, _u2, _u3), _) in self.possible_coalitions[(i, j, k)]):
                low = mid
                continue
            is_feasible, sol = self.determine_feasibility(i, j, k, common_edges, u1, u2, u3)
            if is_feasible:
                utility_vector = (self.utility(i, j, k, i, common_edges, sol), \
                                 self.utility(i, j, k, j, common_edges, sol), \
                                self.utility(i, j, k, k, common_edges, sol))
                low = mid
            else:
                high = mid
        
        if is_feasible:
            possible_coalitions.add((utility_vector, sol))
            # possible_exchanges.add(sol)

        if idx == 0:
            u1, u2, u3 = high, fixed1, fixed2
        elif idx == 1:
            u1, u2, u3 = fixed1, high, fixed2
        else:
            u1, u2, u3 = fixed1, fixed2, high

        is_feasible, sol = self.determine_feasibility(i, j, k, common_edges, u1, u2, u3)
        if is_feasible:
            utility_vector = (self.utility(i, j, k, i, common_edges, sol), \
                                 self.utility(i, j, k, j, common_edges, sol), \
                                self.utility(i, j, k, k, common_edges, sol))
            possible_coalitions.add((utility_vector, sol))
            # possible_exchanges.add(sol)
    
        return possible_coalitions
    # , possible_exchanges
    
    def run(self):
        # num_of_coalitions = 0
        for i in range(self.N):
            for j in range(i+1, self.N):
                for k in range(j+1, self.N):
                    self.possible_coalitions[(i, j, k)] = set()
                    print(i, j, k)
                    path1, path2, path3 = self.paths[i], self.paths[j], self.paths[k]
                    edges_path1 = set((u, v, self.G[u][v]['weight']) for u, v in zip(path1[:-1], path1[1:]))
                    edges_path2 = set((u, v, self.G[u][v]['weight']) for u, v in zip(path2[:-1], path2[1:]))
                    edges_path3 = set((u, v, self.G[u][v]['weight']) for u, v in zip(path3[:-1], path3[1:]))

                    common_edges = dict()
                    common_edges[(i, j)] = edges_path1 & edges_path2
                    common_edges[(i, k)] = edges_path1 & edges_path3
                    common_edges[(j, k)] = edges_path2 & edges_path3

                    if len(common_edges[(i, j)]) == 0 and len(common_edges[(i, k)]) == 0 and len(common_edges[(j, k)]) == 0:
                        continue
                

                    u1_upd = self.utility(i, j, k, i, common_edges, (0, 0, 1, 0, 1, 0))
                    u2_upd = self.utility(i, j, k, j, common_edges, (1, 0, 0, 0, 0, 1))
                    u3_upd = self.utility(i, j, k, k, common_edges, (0, 1, 0, 1, 0, 0))
                    
                    print(u1_upd, u2_upd, u3_upd)
                    
                    if u1_upd == 0 or u2_upd == 0 or u3_upd == 0:
                        continue

                    if u1_upd >= u2_upd and u1_upd >= u3_upd:
                        for u2 in np.arange(u2_upd, 0, -0.1):
                            for u3 in np.arange(u3_upd, 0, -0.1):
                                coalitions = self.binary_search_and_collect(u1_upd, u2, u3, common_edges, i, j, k, 0)
                                self.possible_coalitions[(i, j, k)] |= coalitions
                                # self.possible_exchanges[(i, j, k)] |= exchanges
                    elif u2_upd >= u1_upd and u2_upd >= u3_upd:
                        for u1 in np.arange(u1_upd, 0, -0.1):
                            for u3 in np.arange(u3_upd, 0, -0.1):
                                coalitions = self.binary_search_and_collect(u2_upd, u1, u3, common_edges, i, j, k, 1)
                                self.possible_coalitions[(i, j, k)] |= coalitions
                                # self.possible_exchanges[(i, j, k)] |= exchanges
                    else:
                        for u1 in np.arange(u1_upd, 0, -0.1):
                            for u2 in np.arange(u2_upd, 0, -0.1):
                                coalitions = self.binary_search_and_collect(u3_upd, u1, u2, common_edges, i, j, k, 2)
                                self.possible_coalitions[(i, j, k)] |= coalitions
                                # self.possible_exchanges[(i, j, k)] |= exchanges

                    self.num_of_coalitions += len(self.possible_coalitions[(i, j, k)])

        print(self.num_of_coalitions)

    def export_coalitions(self, replication_id):
        export_data = []
        for i in range(self.N):
            for j in range(i+1, self.N):
                for k in range(j+1, self.N):
                    for (utility_vector, exchange) in self.possible_coalitions[(i, j, k)]:
                        x12, x13, x21, x23, x31, x32 = exchange
                        u1, u2, u3 = utility_vector
                        export_data.append((i, j, k, u1, u2, u3, x12, x13, x21, x23, x31, x32))

        with open("coalition/{}-{}.csv".format(self.N, replication_id), "w", newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(export_data)
        # print(export_data)

if __name__ == "__main__":
    num_of_replication = 1
    num_of_coalitions = 0
    num_of_oracle_invokes = 0
    running_time = 0

    for idx in range(num_of_replication):
        start_time = time.time()
        cpa = CPA('manhatten.graphml', 24)
        cpa.run()
        num_of_coalitions += cpa.num_of_coalitions
        num_of_oracle_invokes += cpa.invoke_oracle_num
        cpa.export_coalitions(idx)
        end_time = time.time()
        running_time += end_time - start_time
        print("Replication {}: Running time: {}".format(idx, end_time - start_time))

    num_of_oracle_invokes /= num_of_replication
    num_of_coalitions /= num_of_replication
    running_time /= num_of_replication

    # print("Number of oracle invokes: ", num_of_oracle_invokes)
    # print("Number of coalitions: ", num_of_coalitions)
    # print("Running time: ", running_time)

    print(num_of_oracle_invokes)
    print(num_of_coalitions)
    print(running_time)
