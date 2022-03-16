
import argparse

#====================================================
# Args for sure used
parser = argparse.ArgumentParser()

# general experiment params 
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--epochs', type=int, default=20)

args = parser.parse_args()

print("First arg: {}".format(args.seed))
print("Second arg: {}".format(args.epochs))