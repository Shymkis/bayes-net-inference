#
# Author: Robert Geraghty 
# Email : rrg053@utulsa.edu
# 

import networkx as nx
from networkx.algorithms.dag import topological_sort
import json
from multiprocessing import Process, Lock, Manager
import time
import ujson
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', default='bn.json', help='The name of the json file from gen-bn')
parser.add_argument('-q', '--query', default='0', help='The query variable name')
parser.add_argument('-m', '--multiprocessing', action='store_true', help='Enables multiprocessing over the query variable values, probably don\'t use this if your other methods don\'t use multiprocessing, as this will run twice as fast.')
args = parser.parse_args()

multi = args.multiprocessing

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
        return 1-prob if evidence[var] == 0 else prob 

    def enumeration_ask(self, query_var, evidence = {}):
        manager = Manager()
        Q = manager.dict()
        possibilities = [0,1]
        if query_var in evidence:
            other_vals = [x for x in possibilities if x != evidence[query_var]]
            out = {evidence[query_var]:1}
            for val in other_vals:
                out.update({val:0})
            return out
        topsort = list(topological_sort(self.net))
        if not multi:
            for x in possibilities:
                print(evidence)
                print('Enumerating with query var value', x)
                e = evidence
                e.update({query_var  :x})
                Q[x] = self.enumerate_all(topsort, e)
            return self.normalize(Q)
        else:
            lock = Lock()
            processes = []
            for x in possibilities:
                print('Enumerating with query var value', x)
                e = json_deep_copy(evidence)
                e.update({query_var  :x})
                processes.append(Process(target=self.enumerate_all, args=(topsort, e, Q, lock, query_var)))
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            return self.normalize(Q)

    def enumerate_all(self, v, ev, Q = None, lock = None, query_var = None):
        evidence = None
        varlist = None
        if lock == None:
            evidence = json_deep_copy(ev)
            varlist = json_deep_copy(v)
        else:
            lock.acquire()
            evidence = json_deep_copy(ev)
            varlist = json_deep_copy(v)
            lock.release()
        if varlist == []:
            if lock == None:
                return 1.0
            lock.acquire()
            Q.update({evidence[query_var]:1.0})
            lock.release()
            return

        Y = varlist[0]
        if Y in evidence:
            prob = self.prob_util(Y, evidence, self.P_x_given(Y, evidence))
            ret = prob * self.enumerate_all(varlist[1:], evidence)
            # print("Probability of {} is {} given {} is {}".format(str(Y), str(evidence[Y]), str(evidence), str(ret)))
            if lock == None:
                return ret
            lock.acquire()
            Q.update({evidence[query_var]:ret})
            lock.release()
            return
        else:
            e = evidence
            sum = 0
            for val in [1,0]:
                e.update({Y: val})
                ret = self.prob_util(Y, e, self.P_x_given(Y, e)) * self.enumerate_all(varlist[1:], e)
                # print("Probability of {} is {} given {} is {}".format(str(Y), str(e[Y]), str(e), str(ret)))
                sum += ret
            if lock == None:
                return sum
            lock.acquire()
            Q.update({evidence[query_var]:sum})
            lock.release()
            return
   
    def P_x_given(self, x, evidence):
        parent_values = []
        for parent in self.net.predecessors(x):
           parent_values.append(evidence[parent])
           
        match = [cp for cp in self.nodes[x].cpt if cp[0] == parent_values]
        return match[0][1]
    
    def normalize(self, distribution):
        # print(distribution)
        s = sum(list(distribution.values()))
        for key in list(distribution.keys()):
            distribution.update({key:distribution[key]/s})
        return distribution


def json_deep_copy(data):
    return ujson.loads(ujson.dumps(data))

class Bayes_Node():
    def __init__(self, name, parents, cpt):
        self.name = name 
        self.parents = parents
        self.cpt = cpt

def runner():
    bn = Bayes_Net()
    bn.create_from_json(args.file)
    # bn.draw()

    starttime = time.time()
    if multi:
        print("Running multiprocessed exact inference on query variable", args.query)
    else:
        print("Running exact inference on query variable", args.query)
    exact_enum = bn.enumeration_ask(args.query)
    # print(exact_enum)
    endtime = time.time()
    exact_time = (endtime-starttime)

    print("-----Finished!-----")
    print("Time\t:", exact_time,
            "secs\nP_False\t:", exact_enum[0],
            "\nP_True\t:",exact_enum[1])
    
if __name__ == "__main__":
    runner()
