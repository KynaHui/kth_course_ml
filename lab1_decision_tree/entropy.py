import dtree
import monkdata as m
import numpy as np
import matplotlib.pyplot as plt


print(dtree.entropy(m.monk1))
print(dtree.entropy(m.monk1test))
print(dtree.entropy(m.monk2))
print(dtree.entropy(m.monk2test))
print(dtree.entropy(m.monk3))
print(dtree.entropy(m.monk3test))
print()

datasets = [m.monk1, m.monk2, m.monk3]
for dataset in datasets:
    for i in range(6):
        print(str(round(dtree.averageGain(dataset, m.attributes[i]),6)) + " & ", end="")
    print()


