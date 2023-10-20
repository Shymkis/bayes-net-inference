#
# Author: Joe Shymanski
# Email:  joe-shymanski@utulsa.edu
# 

import networkx as nx
from networkx.algorithms.dag import topological_sort
import json
import time
import ujson
import argparse
import random
import matplotlib.pyplot as plt
import math

class Bayes_Net():
    def __init__(self):
        self.net = nx.DiGraph()
        self.nodes = {}

    def create_from_json(self, file):
        with open(file) as json_file:
            data = json.load(json_file)
            for name, value in data.items():
                node = Bayes_Node(str(name), [str(i) for i in value['parents']], value['prob'])
                self.nodes.update({str(name): node})
                self.net.add_node(node.name, cpt = node.cpt, color='black')
                for parent in node.parents:
                    self.net.add_edge(parent, node.name, label=(parent+"->"+node.name), color='black')#, minlen=(abs(int(parent)-int(node.name))*1))
    def add_node(self, node):
        self.net.add_node(node.name, cpt = node.cpt)
        for parent in node.parents:
            self.net.add_edge(parent.name, node.name)

    def prob_util(self, var, evidence, prob):
        return prob if evidence[var] else 1-prob 

    def P_x_given_parents(self, x, evidence):
        parent_values = []
        for parent in self.net.predecessors(x):
           parent_values.append(evidence[parent])
        match = [cp for cp in self.nodes[x].cpt if cp[0] == parent_values]
        return match[0][1]
    
    def P_x_given_markov_blanket(self, x, evidence):
        children = [c for c in self.nodes if x in self.net.predecessors(c)]
        p_positive = self.P_x_given_parents(x, evidence)
        p_negative = 1 - p_positive
        for child in children:
            tmp = evidence[x]
            evidence[x] = True
            p_positive *= self.prob_util(child, evidence, self.P_x_given_parents(child, evidence))
            evidence[x] = False
            p_negative *= self.prob_util(child, evidence, self.P_x_given_parents(child, evidence))
            evidence[x] = tmp
        return p_positive / (p_positive + p_negative)
    
    def normalize(self, distribution):
        s = sum(list(distribution.values()))
        for key in list(distribution.keys()):
            distribution.update({key:distribution[key]/s})
        return distribution
    
    def enumeration_ask(self, query_var, evidence = {}):
        Q = {}
        possibilities = [0,1]
        if query_var in evidence:
            other_vals = [x for x in possibilities if x != evidence[query_var]]
            out = {evidence[query_var]:1}
            for val in other_vals:
                out.update({val:0})
            return out
        topsort = list(topological_sort(self.net))
        for x in possibilities:
            print(evidence)
            print('Enumerating with query var value', x)
            e = evidence
            e.update({query_var  :x})
            Q[x] = self.enumerate_all(topsort, e)
        return self.normalize(Q)

    def enumerate_all(self, v, ev):
        evidence = json_deep_copy(ev)
        varlist = json_deep_copy(v)
        if varlist == []:
            return 1.0

        Y = varlist[0]
        if Y in evidence:
            prob = self.prob_util(Y, evidence, self.P_x_given_parents(Y, evidence))
            ret = prob * self.enumerate_all(varlist[1:], evidence)
            # print("Probability of {} is {} given {} is {}".format(str(Y), str(evidence[Y]), str(evidence), str(ret)))
            return ret
        else:
            e = evidence
            sum = 0
            for val in [1,0]:
                e.update({Y: val})
                ret = self.prob_util(Y, e, self.P_x_given_parents(Y, e)) * self.enumerate_all(varlist[1:], e)
                # print("Probability of {} is {} given {} is {}".format(str(Y), str(e[Y]), str(e), str(ret)))
                sum += ret
            return sum

    def prior_sample(self):
        sampled_event = {} # x, an event with n elements
        topsort = list(topological_sort(self.net)) # X_1, ..., X_n
        for X_i in topsort:
            prob = self.P_x_given_parents(X_i, sampled_event) # P(X_i | parents(X_i))
            sampled_val = random.random() < prob # random sample
            sampled_event.update({X_i: sampled_val}) # add sampled value to event
        return sampled_event
    
    def likelihood_weighting(self, X, e, N):
        W = {} # vector of weighted counts for each value of X
        for _ in range(N):
            sampled_event, w = self.weighted_sample(e)
            if sampled_event[X] not in W:
                W[sampled_event[X]] = 0 # counts are initially zero
            W[sampled_event[X]] += w # increment the count for the sampled value of X
        return self.normalize(W)
    
    def weighted_sample(self, e):
        w = 1
        sampled_event = json_deep_copy(e) # x, an event with n elements and values fixed from e
        topsort = list(topological_sort(self.net)) # X_1, ..., X_n
        for X_i in topsort:
            if X_i in sampled_event:
                w *= self.prob_util(X_i, sampled_event, self.P_x_given_parents(X_i, sampled_event)) # P(X_i = x_ij | parents(X_i))
            else:
                prob = self.P_x_given_parents(X_i, sampled_event) # P(X_i | parents(X_i))
                sampled_val = random.random() < prob # random sample using evidence
                sampled_event.update({X_i: sampled_val}) # add sampled value to event
        return sampled_event, w

    def gibbs_sample(self, Z, current_state):
        Z_i = random.choice(Z) # randomly choose Z_i
        prob = self.P_x_given_markov_blanket(Z_i, current_state) # P(Z_i | MB(Z_i))
        current_state[Z_i] = random.random() < prob # random sample using evidence

    def gibbs_ask(self, X, e, N):
        C = {}
        Z = [z for z in topological_sort(self.net) if z not in e]
        current_state = json_deep_copy(e)
        for z in Z:
            current_state[z] = random.random() < .5 # initialize random value
        for _ in range(N):
            self.gibbs_sample(Z, current_state)
            if current_state[X] not in C:
                C[current_state[X]] = 0 # counts are initially zero
            C[current_state[X]] += 1 # increment the count for the sampled value of X
        return self.normalize(C)
    
    def metropolis_hastings(self, X, e, N, p = .95):
        C = {}
        Z = [z for z in topological_sort(self.net) if z not in e]
        current_state, _ = self.weighted_sample(e)
        for _ in range(N):
            if random.random() < p:
                self.gibbs_sample(Z, current_state)
            else:
                current_state, _ = self.weighted_sample(e)
            if current_state[X] not in C:
                C[current_state[X]] = 0 # counts are initially zero
            C[current_state[X]] += 1 # increment the count for the sampled value of X
        return self.normalize(C)


def json_deep_copy(data):
    return ujson.loads(ujson.dumps(data))

class Bayes_Node():
    def __init__(self, name, parents, cpt):
        self.name = name
        self.parents = parents
        self.cpt = cpt

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, default='bn.json', help='The name of the json file from gen-bn')
parser.add_argument('-m', '--method', type=str, choices=['likelihood', 'gibbs', 'mh'], default='likelihood', help='The sampling method')
parser.add_argument('-q', '--query', type=str, default='0', help='The query variable name')
parser.add_argument('-n', '--num_samples', type=int, default=10000, help='Number of samples')
parser.add_argument('-p', '--plot', type=bool, default=False, help='Create plot')
args = parser.parse_args()

def runner():
    bn = Bayes_Net()
    bn.create_from_json(args.file)
    # bn.draw()

    ev = {"0": False, "4": False}

    if args.plot:
        exact = bn.enumeration_ask(args.query, json_deep_copy(ev))
        x, y1, y2, y3, t1, t2, t3 = [], [], [], [], [], [], []
        for n in range(250, 25001, 250):
            x.append(n)

            print("Running likelihood inference on query variable", args.query, "with evidence", ev, "using", n, "samples")
            starttime = time.time()
            approx = bn.metropolis_hastings(args.query, ev, n, p = .95)
            y1.append(abs(approx[True] - exact[1]))
            t1.append(time.time() - starttime)

            print("Running gibbs inference on query variable", args.query, "with evidence", ev, "using", n, "samples")
            starttime = time.time()
            approx = bn.metropolis_hastings(args.query, ev, n, p = .85)
            y2.append(abs(approx[True] - exact[1]))
            t2.append(time.time() - starttime)

            print("Running mh inference on query variable", args.query, "with evidence", ev, "using", n, "samples")
            starttime = time.time()
            approx = bn.metropolis_hastings(args.query, ev, n, p = .75)
            y3.append(abs(approx[True] - exact[1]))
            t3.append(time.time() - starttime)

        plt.plot(x, y1, label="p = .95")
        plt.plot(x, y2, label="p = .85")
        plt.plot(x, y3, label="p = .75")
        plt.xlabel("Number of samples")
        plt.ylabel("Error")
        plt.legend()
        plt.show()

        plt.plot(x, t1, label="p = .95")
        plt.plot(x, t2, label="p = .85")
        plt.plot(x, t3, label="p = .75")
        plt.xlabel("Number of samples")
        plt.ylabel("Time elapsed (s)")
        plt.legend()
        plt.show()
    else:
        print("Running", args.method, "inference on query variable", args.query, "with evidence", ev, "using", args.num_samples, "samples")
        func = bn.likelihood_weighting
        if args.method == "gibbs":
            func = bn.gibbs_ask
        elif args.method == "mh":
            func = bn.metropolis_hastings
        approx = func(args.query, ev, args.num_samples)
        endtime = time.time()
        approx_time = endtime - starttime
        print("-----Finished!-----")
        print("Time\t:", approx_time, "secs\nResult\t:", approx)
    
if __name__ == "__main__":
    runner()
