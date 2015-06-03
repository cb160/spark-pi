#------------------------------------
#
# Extension of PI estimation example
#
#------------------------------------
import argparse
from random import SystemRandom
from time import time
from operator import add
from pyspark import SparkContext

def f(_):
    b = SystemRandom()
    x = b.random() * 2 - 1
    y = b.random() * 2 - 1
    return 1 if x ** 2 + y ** 2 < 1 else 0

# Uses Spark to parallelise estimation of PI using f(), with specified number of calls
# distributed over the specified number of partitions
def test_run_no_persistence(desc, test_num, n, partitions, sc, test_data, results):

    t1 = time()

    count = test_data.map(f).reduce(add)

    t2 = time()

    results.append((desc, test_num, n, count, t2-t1, 4.0 * count / n))

# Uses Spark to parallelise estimation of PI using f(), with specified number of calls
# distributed over the specified number of partitions
def test_run_with_persistence(desc, test_num, n, partitions, persisted_data, results):

    t1 = time()

    count = persisted_data.reduce(add)

    t2 = time()

    results.append((desc, test_num, n, count, t2-t1, 4.0 * count / n))

# Repeatedly perform test, both with and without cached data
def test_over_iterations(desc, iterations_per_partition, partitions, cycles, sc, results):

    n = iterations_per_partition * partitions

    test_data = sc.parallelize(xrange(1, n + 1), partitions)

    # First run - no persistence
    for iter_num in xrange(cycles):
        test_run_no_persistence(desc + " No caching", iter_num, n, partitions, sc, test_data, results)

    # Second run - with persistence
    persisted_data = test_data.map(f).persist()
    persisted_data.reduce(add) # Forces creation of persisted data
    
    for iter_num in xrange(cycles):
        test_run_with_persistence(desc + " With caching", iter_num, n, partitions, persisted_data, results)

# Execute the test with the specified number of cycles in each test, across the
# specified number of Spark partitions
def run(cycles, partitions):
    results = []

    sc = SparkContext(appName="PythonPi")

    for iters_per_partition in [100000]:
        test_over_iterations("", iters_per_partition, partitions, cycles, sc, results)

    sc.stop()

    for (desc, iter_num, num, count, secs, pi) in results:
        print('{0}-{1}: {2} generated {3} in {4} secs (PI = {5})'.format(desc, iter_num, num, count, secs, pi))    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Estimate PI')
    parser.add_argument('-cycles',type=int, default=5, help='Number of test cycles')
    parser.add_argument('-partitions', type=int, default=8, help='The number of Spark partitions to use')
    args = parser.parse_args()

    run(args.cycles, args.partitions)

