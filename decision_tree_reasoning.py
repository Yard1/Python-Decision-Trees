#############################
###
### Decision Tree Reasoning
### Requires numpy and pandas pip packages to be installed. More info on installing packages: https://docs.python.org/3/installing/index.html
### Written in Python 3.6
###
### Copyright (c) 2019 Antoni Baum (Yard1)
### Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
### The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
### THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
###
### usage: decision_tree_reasoning.py <command> [<args>]
###
### Commands:
### create      Create a new ruleset (saved to ruleset.pickle) using a .csv file (create <input filename>)
### forward     Forward reasoning
### backward    Backward reasoning for a given conclusion value (backward <conclusion>)
###
### Decision Tree reasoning
###
### positional arguments:
###   command     Subcommand to run
###
### optional arguments:
###   -h, --help  show this help message and exit
###
#############################

import csv
import codecs
import sys
import collections
import math
import numpy as np
import pandas
import argparse
import pickle
import itertools

def load_data(file):
    data = {}
    conclusion = ""
    with open(file, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row[0] not in data:
                data[row[0]] = [0]*(len(row)-2)
            for i, col in enumerate(row[2:]):
                if str(col) == "1":
                    data[row[0]][i] = str(row[1])
            conclusion = row[0]
    return (data, conclusion)

def calculate_entropy_at_node(df, conclusion):
    entropy = 0
    values = df[conclusion].unique()
    for value in values:
        entropy += calculate_entropy(df[conclusion].value_counts()[value], len(df[conclusion]))
    return entropy

def calculate_entropy_for_attribute(df, attribute, conclusion):
    entropy_total = 0
    conclusion_values = df[conclusion].unique()
    attribute_values = df[attribute].unique()
    for attribute_value in attribute_values:
        entropy = 0
        for conclusion_value in conclusion_values:
            ni = len(df[attribute][df[attribute] == attribute_value][df[conclusion] == conclusion_value])
            n = len(df[attribute][df[attribute] == attribute_value])
            entropy += calculate_entropy(ni, n)
        entropy_total += (n/len(df)) * entropy
    return entropy_total

def calculate_entropy(a, b):
    if a == 0 or b == 0:
        return 0
    return -((a/b)*math.log2(a/b))

def find_winner(df, conclusion):
    ig = []
    for key in df.keys():
        if key != conclusion:
            ig.append(calculate_entropy_at_node(df,conclusion)-calculate_entropy_for_attribute(df,key,conclusion))
    return df.keys()[np.argmax(ig)]
  
  
def get_subtable(df, node, value):
  return df[df[node] == value].reset_index(drop=True)

def build_tree(df, conclusion, tree=None):   
    node = find_winner(df, conclusion)
    
    attValue = np.unique(df[node])

    if not tree:
        tree={}
        tree[node] = {}
    for value in attValue:
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable[conclusion],return_counts=True)
        
        if len(counts)==1:
            tree[node][value] = clValue[0]
        else:        
            tree[node][value] = build_tree(subtable, conclusion)

    return tree

def walkDict(aDict, results, path=()):
    for  k in aDict:
        if type(aDict[k]) != dict:
            t = list(path[:-1])
            t = zip(t[0::2],t[1::2])
            results[aDict[k].lower()].add(tuple(t))
            pass
        else:
            walkDict(aDict[k], results, path+(k.lower(),))

def backward_reasoning(tree, goal, possible_values, conc):
    results = collections.defaultdict(set)
    not_checked = set(possible_values.keys())
    not_checked.remove(conc)
    walkDict(tree, results)
    print("Backward reasoning for \"" + goal + "\":")
    results = results[goal]
    while len(results) > 1:
        var = None
        while True:
            var2 = None
            var = str(input("Please enter the predicate (\"%s\"): "  % "\", \"".join(not_checked)))
            var = var.strip().lower()
            if var not in possible_values.keys():
                print("\"" + var + "\" is not a valid predicate")
            elif var not in not_checked:
                print("\"" + var + "\" has already been checked")
            else:
                while True:
                    var2 = str(input("Please enter the predicate value (\"%s\"): " % "\", \"".join(possible_values[var])))
                    var2 = var2.strip().lower()
                    if var2 not in possible_values[var]:
                        print("\"" + var2 + "\" is not a valid predicate value")
                    else:
                        break
                break
        not_checked.remove(var)
        new_results = {x for x in results if (var.lower(), var2.lower()) in x}
        if len(new_results) == 0:
            return False
        results = new_results
        print()

    return True


def forward_reasoning(tree, possible_values):
    print("Forward reasoning:")
    result = ""
    current_node = tree
    current_key = list(tree.keys())[0]
    while True:
        available_keys = current_node[current_key].keys()
        print()
        print("Predicate: \"" + current_key + "\"")
        print("Possible values: \"" + "\", \"".join(available_keys) + "\"")
        var = ""
        while True:
            var = str(input("Please enter the chosen predicate value: ")).lower()
            if not var in available_keys:
                print("\"" + var + "\" is not a valid predicate value")
                var = None
            else:
                break
        if isinstance(current_node[current_key][var], dict):
            new_key = list(current_node[current_key][var].keys())[0]
            current_node = current_node[current_key][var]
            current_key = new_key
        else:
            result = current_node[current_key][var]
            break
    return result

def get_possible_values(df):
    uniques = {}
    for x in df:
        st = df[x].unique()
        uniques[x] = { x.lower() for x in st }
    return uniques

def create():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args(sys.argv[2:])
    d = load_data(args.filename)
    df = pandas.DataFrame(d[0])
    df.columns = map(str.lower, df.columns)
    df = df.applymap(str.lower)
    n = build_tree(df, d[1].lower())
    possible_values = get_possible_values(df)
    with open('ruleset.pickle', 'wb') as handle:
        pickle.dump((n, possible_values, d[1].lower()), handle, pickle.HIGHEST_PROTOCOL)
    print("Saved ruleset to ruleset.pickle")
    print("Possible conclusions are: \"%s\"" % "\", \"".join(possible_values[d[1].lower()]))

def backward():
    parser = argparse.ArgumentParser()
    parser.add_argument('conclusion')
    args = parser.parse_args(sys.argv[2:])
    goal = args.conclusion
    with open('ruleset.pickle', 'rb') as handle:
        n = pickle.load(handle)
    tree = n[0]
    possible_values = n[1]
    conclusion = n[2]

    if not goal.lower() in possible_values[conclusion]:
        print("\"" + goal + "\" is not a valid conclusion")
        return

    result = backward_reasoning(tree, goal.lower(), possible_values, conclusion)
    print()
    if result:
        print("\"" + goal + "\" is a valid conclusion for given predicates")
    else:
        print("\"" + goal + "\" is NOT a valid conclusion for given predicates")

def forward():
    with open('ruleset.pickle', 'rb') as handle:
        n = pickle.load(handle)
    tree = n[0]
    possible_values = n[1]
    conclusion = n[2]

    result = forward_reasoning(tree, possible_values)
    print()
    print("\"" + result + "\" is the conclusion for given predicates")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Decision Tree reasoning',
        usage='''decision_tree_reasoning.py <command> [<args>]

Commands:
create      Create a new ruleset (saved to ruleset.pickle) using a .csv file (create <input filename>)
forward     Forward reasoning
backward    Backward reasoning for a given conclusion value (backward <conclusion>)
''')
    parser.add_argument('command', help='Subcommand to run')
    args = parser.parse_args(sys.argv[1:2])
    if not (args.command == 'create' or args.command == 'backward' or args.command == 'forward'):
        print('Unrecognized command')
        parser.print_help()
        exit(1)
    if args.command == 'create':
        create()
    elif args.command == 'backward':
        backward()
    elif args.command == 'forward':
        forward()