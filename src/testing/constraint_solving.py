from z3 import *
from typing import Any
# import numpy as np 
# import tensorflow as tf

uP = ['p1', 'p2']
constraints1 = {
    'p1': {
        'type': 'tensor',
        'rank': [(1, 4),],
        'shape': None, 
        'dtype': 'float32',
        'value': None,
    },
    'p2': {
        'type': 'int',
        'value': [(-1, 3),],
    }
}

constraints2 = {
    'p1': {
        'type': 'tensor',
        'rank': [(1, 10),],
        'shape': None,
        'dtype': 'float32',
        'value': None,
    },
    'p2': {
        'type': 'int',
        'value': [(-1, 3),],
    }
}

class Tensor():
    def __init__(self, name, rank=None, shape=None, dtype=None):
        self.name = name
        self.rank = Int(name+'_rank')
        self.shape = shape
        self.dtype = dtype
        self.value = Int(name+'_value')
        
    def __repr__(self):
        return f"name: {self.name}, rank={self.rank}, shape={self.shape}, dtype={self.dtype}, value={self.value}"

class FeasibleSet():
    def __init__(self, name, fs_type):
        self.name = name
        self.fs_type = fs_type
        self.solutions = []
    
    def __repr__(self):
        return f"fs({self.fs_type}):{self.solutions}"

def all_smt(s, initial_terms):
    def block_term(s, m, t):
        s.add(t != m.eval(t, model_completion=True))
    def fix_term(s, m, t):
        s.add(t == m.eval(t, model_completion=True))
    def all_smt_rec(terms):
        if sat == s.check():
           m = s.model()
           yield m
           for i in range(len(terms)):
               s.push()
               block_term(s, m, terms[i])
               for j in range(i):
                   fix_term(s, m, terms[j])
               yield from all_smt_rec(terms[i:])
               s.pop()
        else:
            #print('No solution found!')
            return None
    yield from all_smt_rec(list(initial_terms))
    
def get_fs(s, varlist, t):
    obj = all_smt(s, varlist)
    fs = FeasibleSet(varlist, fs_type=t)
    times = 5
    for i in range(times):
        try:
            new_ans = next(obj)
            fs.solutions.append(new_ans)
        except StopIteration:
            break
    return fs

def solve_joint_constraints(c1, c2, var):
    intersect_solver = Solver()
    
    intersect_solver.add(Or([And(var>i[0], var<i[1]) for i in c1]))
    intersect_solver.add(Or([And(var>i[0], var<i[1]) for i in c2]))
    
    intersect_fs = get_fs(intersect_solver, [var], 'intersect')
    
    da_solver = Solver()
    da_solver.add(Or([And(var>i[0], var<i[1]) for i in c1]))
    da_solver.add(Not(Or([And(var>i[0], var<i[1]) for i in c2])))
    
    db_solver = Solver()
    db_solver.add(Not(Or([And(var>i[0], var<i[1]) for i in c1])))
    db_solver.add(Or([And(var>i[0], var<i[1]) for i in c2]))
    
    da_fs = get_fs(da_solver, [var], 'dA')
    db_fs = get_fs(db_solver, [var], 'dB')
    return [intersect_fs, da_fs, db_fs]

def joint_analysis():
    
    res = {}
    for p in uP:
        
        if constraints1[p]['type'] == 'tensor' and constraints2[p]['type'] == 'tensor':
            v = Tensor(p)
            v.dtype = constraints1[p]['dtype']
            if constraints1[p]['rank'] and constraints1[p]['rank']:
                new_rank_intersect = solve_joint_constraints(constraints1[p]['rank'], constraints2[p]['rank'], v.rank)
                v.rank = new_rank_intersect
            if constraints1[p]['shape'] and constraints2[p]['shape']:
                pass
            if constraints1[p]['value'] and constraints2[p]['value']:
                print("Y")
                new_value_intersect = solve_joint_constraints(constraints1[p]['value'], constraints2[p]['value'], v.value)
                v.value = new_value_intersect
            
            v.shape = None # TODO
           
        elif constraints1[p]['type'] == 'int':
            v = Int(p)
            if constraints1[p]['value']:
                new_value_intersect = solve_joint_constraints(constraints1[p]['value'], constraints2[p]['value'], v)
                v = new_value_intersect
        res[p] = v
    return res

    # domain = IntSort()
    # range = BoolSort()
    
    # A = Function('A', domain, range)
    # B = Function('B', domain, range)
    # C = Function('C', domain, range)
    # D = Function('D', domain, range)
    
    # x = Int('x')
    # s.add(ForAll(x, C(x) == And(A(x), B(x))))
    # s.add(ForAll(x, D(x) == And(A(x), Not(B(x)))))
    # s.add(ForAll(x, A(x) == (x > 0)))
    # s.add(ForAll(x, B(x) == (x % 2 == 0)))
    # s.check(C(3))

# p1 = tf.constant(np.random.rand(*solution['p1'].rank).astype(np.float32)) 
# p2 = solution['p2']

if __name__ == '__main__':
    
    old = {"p1":Tensor("p1"), "p2":None}
    times = 1
    for t in range(times):
        res = joint_analysis()
        print(f"Joint Analysis Results: {res}")