import math
import random

from torch.utils.data import Sampler


class TokenBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: list[int],
        max_tokens: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rng = random.Random(self.seed)
    
    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            self.rng.shuffle(indices)
        
        batch = []
        total = 0
        for idx in indices:
            length = self.lengths[idx]
            if total + length > self.max_tokens and batch:
                yield batch
                batch = []
                total = 0
            batch.append(idx)
            total += length
        
        if batch and not self.drop_last:
            yield batch
    
    def __len__(self):
        if not self.lengths:
            return 0
        total_tokens = sum(self.lengths)
        return math.ceil(total_tokens / self.max_tokens)


class BalancedLengthSampler(Sampler[list[int]]):
    """
    將資料按長度排序後切分為 batch_size 個桶子，
    每個 batch 從每個桶子各抽一個樣本，確保長短混合。
    """
    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rng = random.Random(self.seed)

    def __iter__(self):
        n = len(self.lengths)
        # 取得排序後的索引，改為 reverse=True 讓長檔案排在前面
        indices = list(range(n))
        indices.sort(key=lambda i: self.lengths[i], reverse=True)

        # 計算每個區段（桶子）的大小
        num_batches = n // self.batch_size
        if not self.drop_last and n % self.batch_size != 0:
            num_batches += 1

        # 將索引切分為 batch_size 個桶子
        buckets = []
        for i in range(self.batch_size):
            start = i * num_batches
            end = min(start + num_batches, n)
            bucket = indices[start:end]
            if self.shuffle:
                self.rng.shuffle(bucket)
            buckets.append(bucket)

        # 組合 batch
        for j in range(num_batches):
            batch = []
            for i in range(self.batch_size):
                if j < len(buckets[i]):
                    batch.append(buckets[i][j])

            if len(batch) == self.batch_size:
                yield batch
            elif not self.drop_last and len(batch) > 0:
                yield batch

    def __len__(self):
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        else:
            return (n + self.batch_size - 1) // self.batch_size


class PaddingBatchSampler(Sampler[list[int]]):
    """
    動態 Batch Sampler，考慮 Padding 後的總面積 (Batch Size * Max Length in Batch)。
    這對於精確控制 VRAM 非常有效。
    """
    def __init__(
        self,
        lengths: list[int],
        max_tokens: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rng = random.Random(self.seed)

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            self.rng.shuffle(indices)

        batch = []
        max_len = 0
        for idx in indices:
            length = self.lengths[idx]
            new_max_len = max(max_len, length)
            # 計算如果加入這個樣本，Padding 後的總面積
            if (len(batch) + 1) * new_max_len > self.max_tokens and batch:
                yield batch
                batch = [idx]
                max_len = length
            else:
                batch.append(idx)
                max_len = new_max_len

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if not self.lengths:
            return 0
        # 估計長度（由於是動態的，精確長度難以預先得知，這裡提供一個保守估計）
        total_tokens = sum(self.lengths)
        return math.ceil(total_tokens / self.max_tokens)
