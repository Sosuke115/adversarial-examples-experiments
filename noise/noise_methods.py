"""
https://github.com/ybisk/charNMT-noise
"""

import os, sys, random, yaml
import numpy as np
import click


"""
  Scrambling functions
"""
def swap(w, data, probability=1.0):
  """
    Random swap two letters in the middle of a word
  """
  if random.random() > probability:
    return w
  if len(w) > 3:
    w = list(w)
    i = random.randint(1, len(w) - 3)
    w[i], w[i+1] = w[i+1], w[i]
    return ''.join(w)
  else:
    return w

def random_middle(w):
  """
    Randomly permute the middle of a word (all but first and last char)
  """
  if len(w) > 3:
    middle = list(w[1:len(w)-1])
    random.shuffle(middle)
    middle = ''.join(middle)
    return w[0] + middle + w[len(w) - 1]
  else:
    return w

def fully_random(w, percentage=1.0):
  if random.random() > percentage:
    return w
  """
    Completely random permutation
  """
  w = list(w)
  random.shuffle(w)
  return ''.join(w)

NN = {}
"""
for line in open("noise/" + config["lang"] + ".key"):
  line = line.split()
  NN[line[0]] = line[1:]
"""

def key(w, NN, probability=1.0):
  if random.random() > probability:
    return w
  """
    Swaps $n$ letters with their nearest keys
  """
  w = list(w)
  i = random.randint(0, len(w) - 1)
  char = w[i]
  caps = char.isupper()
  if char in NN:
    w[i] = NN[char.lower()][random.randint(0, len(NN[char.lower()]) - 1)]
    if caps:
      w[i].upper()
  return ''.join(w)

"""
typos = {}
for line in open("noise/" + config["lang"] + ".natural"):
  line = line.strip().split()
  typos[line[0]] = line[1:]
"""


def natural(w, typos, precentage=1.0):
  if random.random() > precentage:
    return w
  if w in typos:
    return typos[w][random.randint(0, len(typos[w]) - 1)]
  return w
