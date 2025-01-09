import numpy as np
from numba.experimental import jitclass
from numba import uint8, uint16, int16


spec = {
    'diffs1': int16[:],
    'diffs2': int16[:],
    'sums': uint8[:],
    'weighted_sums': uint16[:],
    'log_sums': uint8[:],
}


@jitclass(spec)
class Estimate:
    def __init__(self):
        self.diffs1, self.diffs2, self.sums, self.weighted_sums, self.log_sums = self.calculates_for_evaluate()

    def calculates_for_evaluate(self):
        diffs1 = np.empty(65536, dtype=np.int16)
        diffs2 = np.empty(65536, dtype=np.int16)
        sums = np.empty(65536, dtype=np.uint8)
        weighted_sums = np.empty(65536, dtype=np.uint16)
        log_sums = np.empty(65536, dtype=np.uint8)
        # 生成所有可能的行
        for i in range(65536):
            line = [(i // (16 ** j)) % 16 for j in range(4)]
            line = [1 << k if k else 0 for k in line]
            diffs1[i], sums[i], weighted_sums[i], log_sums[i] = self.diffs_evaluation_func(line)
            diffs2[i], sums[i], weighted_sums[i], log_sums[i] = self.diffs_evaluation_func(line[::-1])
        return diffs1, diffs2, sums, weighted_sums, log_sums

    @staticmethod
    def diffs_evaluation_func(line):
        mask_num = [0, 32, 48, 60, 64, 68, 72, 76, 80, 80]
        line_masked = [min(128 + mask_num[int(np.log2(max(k >> 7, 1)))], k) for k in line]
        score = line_masked[0]
        for x in range(3):
            if line_masked[x + 1] > line_masked[x]:
                if x == 0 and line_masked[x] == 0:
                    score += max(line_masked[x + 1], line_masked[x])
                elif line_masked[x] == 128:
                    pass
                elif line_masked[x] < 160:
                    score -= (line_masked[x + 1] - line_masked[x]) << 3
                else:
                    score += min(line_masked[x + 1], line_masked[x])
            elif x < 2:
                score += line_masked[x + 1] + line_masked[x]
            else:
                score += int((line_masked[x + 1] + line_masked[x]) * 0.75)
        line_masked_weighted = [i * np.sqrt(i) for i in line_masked]
        line_masked_log = [np.log(i + 1) for i in line_masked]
        return np.int32(score / 4), sum(line_masked) // 4, sum(line_masked_weighted) // 4, sum(line_masked_log)

    def evaluate(self, s):
        s_reverse = self.reverse(s)
        diffv1, diffv2, diffh1, diffh2 = np.int32(0), np.int32(0), np.int32(0), np.int32(0)
        sums, weighted_sums, log_sums, min_sumv, min_sumh = np.int32(0), np.int32(0), np.int32(0), np.int32(10000), np.int32(10000)
        for i in range(4):
            l1 = (s >> np.uint64(16 * i)) & np.uint64(0xFFFF)
            l2 = (s_reverse >> np.uint64(16 * i)) & np.uint64(0xFFFF)
            diffh1 += self.diffs1[l1]
            diffh2 += self.diffs2[l1]
            diffv1 += self.diffs1[l2]
            diffv2 += self.diffs2[l2]
            sums += self.sums[l1]
            log_sums += self.log_sums[l1]
            weighted_sums += self.weighted_sums[l1]
            min_sumh = min(min_sumh, self.sums[l1])
            min_sumv = min(min_sumv, self.sums[l2])
        if (min_sumv + min_sumh) / sums < 0.2:  # t
            return 404
        if weighted_sums / sums + 50 - log_sums > 13:
            return 404
        else:
            return max(diffv1, diffv2) + max(diffh1, diffh2)

    def is_t(self, s):
        s_reverse = self.reverse(s)
        sums, min_sumv, min_sumh = np.int32(0), np.int32(10000), np.int32(10000)
        for i in range(4):
            l1 = (s >> np.uint64(16 * i)) & np.uint64(0xFFFF)
            l2 = (s_reverse >> np.uint64(16 * i)) & np.uint64(0xFFFF)
            sums += self.sums[l1]
            min_sumh = min(min_sumh, self.sums[l1])
            min_sumv = min(min_sumv, self.sums[l2])
        return min_sumv, min_sumh,sums,(min_sumv + min_sumh) / sums

    @staticmethod
    def reverse(board: np.uint64) -> np.uint64:
        board = (board & np.uint64(0xFF00FF0000FF00FF)) | ((board & np.uint64(0x00FF00FF00000000)) >> np.uint64(24)) | (
                (board & np.uint64(0x00000000FF00FF00)) << np.uint64(24))
        board = (board & np.uint64(0xF0F00F0FF0F00F0F)) | ((board & np.uint64(0x0F0F00000F0F0000)) >> np.uint64(12)) | (
                (board & np.uint64(0x0000F0F00000F0F0)) << np.uint64(12))
        return board


est=Estimate()
