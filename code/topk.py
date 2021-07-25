

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
# Time:  O(n) ~ O(n^2), O(n) on average.
# Space: O(n)
class QuickSelect(object):
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """
        counts = defaultdict(int)
        for ws in words:
            for w in ws:
                counts[w] += 1
        p = []
        for key, val in counts.items():
            p.append((-val, key))
        self.kthElement(p, k)

        result = []
        sorted_p = sorted(p[:k])
        for i in range(k):
            result.append(sorted_p[i][1])
        return result