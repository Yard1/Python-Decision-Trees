#############################
###
### Binary Decision Tree creator
### Requires graphviz to be installed. http://www.graphviz.org
### Requires lolviz, numpy and pandas pip packages to be installed. More info on installing packages: https://docs.python.org/3/installing/index.html
### Written in Python 3.6
###
### Copyright (c) 2019 Antoni Baum (Yard1)
### Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
### The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
### THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
###
### usage: binary_decision_tree.py [-h] file
###
### Create a binary decision tree.
###
### positional arguments:
###  file        Path to the data in .csv format
###  output      Name of output file
###
### optional arguments:
###  -h, --help  show this help message and exit
###
#############################

import csv
import collections
import math
import argparse
import numpy as np
import pandas
import lolviz

def load_data(file):
    data = {}
    conclusion = ""
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if (row[0],row[1]) not in data:
                data[(row[0],row[1])] = [0]*(len(row)-2)
            for i, col in enumerate(row[2:]):
                data[(row[0],row[1])][i] = str(col) == "1"
            conclusion = row[0]
    return (data, conclusion)

def calculate_entropy(a, b):
    if a == 0 or b == 0:
        return 0
    return -((a/b)*math.log2(a/b))

def calculate_entropy_at_node(df, conclusion):
    entropy = 0
    cols = df[conclusion].keys()
    n = len(df[conclusion])
    for c in cols:
        entropy += calculate_entropy(sum(df[conclusion][c]), n)
    return entropy

def calculate_entropy_of_attribute(df, attribute, conclusion, node_entropy):
    total_entropy = {}
    attributes = df[attribute].keys()
    conclusions = df[conclusion].keys()
    for a in attributes:
        n_pos = 0
        n_neg = 0
        entropy_pos = 0
        entropy_neg = 0
        for c in conclusions:
            n_pos += len(df[df[attribute][a]==True][df[conclusion][c]==True])
            n_neg += len(df[df[attribute][a]==False][df[conclusion][c]==True])
        for c in conclusions:
            entropy_pos += calculate_entropy(len(df[df[attribute][a]==True][df[conclusion][c]==True]), n_pos)
            entropy_neg += calculate_entropy(len(df[df[attribute][a]==False][df[conclusion][c]==True]), n_neg)
        total_entropy[a] = node_entropy-((n_pos/(n_pos+n_neg))*entropy_pos+(n_neg/(n_pos+n_neg))*entropy_neg)
    return total_entropy

def find_best(df, conclusion):
    entropies = {}
    checked_keys = set()
    I = calculate_entropy_at_node(df, conclusion)
    attributes = df.keys()
    for k, v in attributes:
        if conclusion != k and k not in checked_keys:
            checked_keys.add(k)
            entropies[k] = calculate_entropy_of_attribute(df, k, conclusion, I)
    max = None
    for k, v in entropies.items():
        for k_2, v_2 in v.items():
            if (not max or max[1] < abs(v_2)):
                max = ((k, k_2), abs(v_2))
    if max[1] == 0:
        for k, v in attributes:
            if conclusion != k and len(df[k][v].unique())>1:
                max = ((k,v),0)
                
    return max[0]

def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)

def check_purity(df, conclusion):
    for c in df[conclusion].keys():
        if sum(df[conclusion][c]) == len(df[conclusion][c]):
            return c
    return False

def build_tree(df, conclusion, tree=None):
    print(df)
    node = find_best(df, conclusion)
    
    if not tree:
        tree = {}
        tree[node] = {}

    for value in [True, False]:
        subtable = get_subtable(df, node, value)

        if subtable.empty:
            break

        purity = check_purity(subtable, conclusion)

        if purity:
            tree[node][value] = purity
        else:        
            tree[node][value] = build_tree(subtable, conclusion)

    return tree

def main(fname, output):
    d = load_data(fname)
    df = pandas.DataFrame(d[0])
    print(df)
    tree = build_tree(df, d[1])
    import pprint
    pprint.pprint(tree)
    g = lolviz.treeviz(tree)
    g.view(filename=output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a binary decision tree.')
    parser.add_argument( 'file', help='Path to the data in .csv format')
    parser.add_argument( 'output', help='Name of output file')
    args = parser.parse_args()
    main(args.file, args.output)