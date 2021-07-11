

from collections import defaultdict
from random import randint

# Bucket Sort
# Time:  O(n + klogk) ~ O(n + nlogn)
# Space: O(n)
class BucketSort(object):
    def topKFrequent(self, words, k):
        counts = defaultdict(int)
        for ws in words:
            for w in ws:
                counts[w] += 1

        buckets = [[]] * (sum(counts.values()) + 1)