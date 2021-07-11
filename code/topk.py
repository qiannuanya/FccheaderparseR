

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
        for i, count in counts.items():
            buckets[count].append(i)

        result = []
        # result_append = result.append
        for i in reversed(range(len(buckets))):
            for j in range(len(buckets[i])):
                # slower
                # result_append(buckets[i][j])
                result.append(buckets[i][j])
                if len(result) == k:
                    return result
        return result


# Quick Select