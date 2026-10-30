"""测试用的假向量模型：可以精确指定两段文本之间的余弦相似度。"""

import hashlib

import numpy as np
from langchain_core.embeddings import Embeddings

DIM = 1536


def _unit(seed: str) -> np.ndarray:
    rng = np.random.default_rng(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16))
    v = rng.normal(size=DIM)
    return v / np.linalg.norm(v)


class ControlledEmbeddings(Embeddings):
    """set(text, anchor, cos)：让 text 的向量和 anchor 的向量余弦相似度恰好为 cos。
    没设置过的文本用各自的随机向量（1536 维随机向量两两之间的相似度接近 0）。"""

    def __init__(self):
        self.fixed: dict[str, np.ndarray] = {}

    def set(self, text: str, anchor: str, cos: float):
        a = self._vec(anchor)
        noise = _unit("noise:" + text)
        noise = noise - noise.dot(a) * a                     # 与 anchor 正交的分量
        noise /= np.linalg.norm(noise)
        self.fixed[text] = cos * a + np.sqrt(1 - cos ** 2) * noise

    def _vec(self, text: str) -> np.ndarray:
        return self.fixed.get(text, _unit(text))

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]
