import math
import random
from collections import defaultdict

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
        """Initialize a sampler that limits each batch by total sequence length."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rng = random.Random(self.seed)

    def __iter__(self):
        """Yield token-budgeted batches of dataset indices."""
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
        """Estimate the number of token-budgeted batches."""
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
        """Initialize a sampler that mixes short and long samples in each batch."""
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rng = random.Random(self.seed)

    def __iter__(self):
        """Yield batches assembled from length-sorted buckets."""
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
        """Return the number of balanced batches."""
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        else:
            return (n + self.batch_size - 1) // self.batch_size


class SmoothedClassBatchSampler(Sampler[list[int]]):
    """
    Smooth the original class distribution and draw fixed-size mini-batches.

    Samples are drawn without replacement within one pass. The pools are rebuilt
    the next time __iter__ is called, which matches the usual epoch reset.
    """

    def __init__(
        self,
        labels: list[int],
        batch_size: int = 500,
        smoothing_power: float = 0.5,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        """
        Initialize a class-balanced batch sampler.

        smoothing_power controls how much the class distribution is flattened:
        1.0 keeps the original distribution, 0.0 makes classes uniform.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not 0.0 <= smoothing_power <= 1.0:
            raise ValueError("smoothing_power must be between 0.0 and 1.0")
        if not labels:
            raise ValueError("labels must not be empty")

        self.labels = labels
        self.batch_size = batch_size
        self.smoothing_power = smoothing_power
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rng = random.Random(self.seed)

        self.class_to_indices = self._build_class_to_indices()
        self.classes = sorted(self.class_to_indices)
        self.class_probs = self._build_smoothed_probs()

    def _build_class_to_indices(self) -> dict[int, list[int]]:
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            if label is None:
                raise ValueError(f"labels[{idx}] is None")
            class_to_indices[label].append(idx)
        return dict(class_to_indices)

    def _build_smoothed_probs(self) -> dict[int, float]:
        counts = {
            label: len(indices)
            for label, indices in self.class_to_indices.items()
        }
        weights = {
            label: count ** self.smoothing_power
            for label, count in counts.items()
        }
        total_weight = sum(weights.values())
        return {
            label: weight / total_weight
            for label, weight in weights.items()
        }

    def _batch_quotas(self, batch_size: int, pools: dict[int, list[int]]) -> dict[int, int]:
        available_classes = [label for label in self.classes if pools[label]]
        if not available_classes:
            return {}

        total_prob = sum(self.class_probs[label] for label in available_classes)
        raw_quotas = {
            label: batch_size * self.class_probs[label] / total_prob
            for label in available_classes
        }
        quotas = {
            label: min(math.floor(raw_quota), len(pools[label]))
            for label, raw_quota in raw_quotas.items()
        }

        remaining = batch_size - sum(quotas.values())
        if remaining <= 0:
            return quotas

        # Prefer classes whose quota had the largest fractional remainder, then
        # keep cycling through classes with remaining samples until the batch is full.
        fill_order = sorted(
            available_classes,
            key=lambda label: raw_quotas[label] - math.floor(raw_quotas[label]),
            reverse=True,
        )
        while remaining > 0:
            filled = False
            for label in fill_order:
                if quotas[label] >= len(pools[label]):
                    continue
                quotas[label] += 1
                remaining -= 1
                filled = True
                if remaining == 0:
                    break
            if not filled:
                break

        return quotas

    def __iter__(self):
        """Yield smoothed-distribution batches without replacement."""
        pools = {
            label: indices.copy()
            for label, indices in self.class_to_indices.items()
        }
        if self.shuffle:
            for indices in pools.values():
                self.rng.shuffle(indices)

        remaining_samples = len(self.labels)
        while remaining_samples > 0:
            current_batch_size = min(self.batch_size, remaining_samples)
            if self.drop_last and current_batch_size < self.batch_size:
                break

            quotas = self._batch_quotas(current_batch_size, pools)
            batch = []
            for label in self.classes:
                quota = quotas.get(label, 0)
                if quota <= 0:
                    continue
                batch.extend(pools[label][-quota:])
                del pools[label][-quota:]

            if self.shuffle:
                self.rng.shuffle(batch)
            if batch:
                remaining_samples -= len(batch)
                yield batch
            else:
                break

    def __len__(self):
        """Return the number of fixed-size class-smoothed batches."""
        if self.drop_last:
            return len(self.labels) // self.batch_size
        return math.ceil(len(self.labels) / self.batch_size)


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
        """Initialize a sampler constrained by padded batch area."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rng = random.Random(self.seed)

    def __iter__(self):
        """Yield batches whose padded area stays within the token budget."""
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
        """Estimate the number of padded-area batches."""
        if not self.lengths:
            return 0
        # 估計長度（由於是動態的，精確長度難以預先得知，這裡提供一個保守估計）
        total_tokens = sum(self.lengths)
        return math.ceil(total_tokens / self.max_tokens)
