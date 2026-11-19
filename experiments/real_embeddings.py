"""Optional CPU text controls. Run from the release root with the text extra installed."""

import argparse
import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity, NearestNeighbors
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from alfm_bem.bem import BidirectionalExperienceMemory
from .metrics import auroc, write_json
from .real_backbone_experiment import authored_fixture


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
SCIENCE = ("sci.space", "sci.med", "sci.electronics")
SPORTS = ("rec.sport.hockey", "rec.sport.baseball")


def keyword_projection(raw, train_fail, train_other):
    """Fit the original full-batch PyTorch objective on a balanced training subset."""
    torch.manual_seed(42)
    network = nn.Sequential(nn.Linear(raw.shape[1], 256), nn.ReLU(),
                            nn.Linear(256, 64), nn.LayerNorm(64))
    optim = torch.optim.Adam(network.parameters(), lr=0.001)
    data = torch.from_numpy(raw)
    for _ in range(20):
        fail, other = network(data[train_fail]), network(data[train_other])
        pair = fail.roll(1, dims=0)
        positive = nn.functional.pairwise_distance(fail, pair)
        negative = nn.functional.pairwise_distance(fail, other)
        loss = positive.square().mean() + (1.0 - negative).relu().square().mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad():
        return network(data).numpy()


def unseen_texts(texts, training, query_groups):
    """Remove blank, train-seen and repeated bodies, preserving query order."""
    seen = {texts[i].strip() for i in training} | {""}
    groups = []
    for indices in query_groups:
        keep = []
        for i in indices:
            body = texts[i].strip()
            if body not in seen:
                keep.append(i)
                seen.add(body)
        groups.append(np.array(keep, dtype=int))
    return groups


def retrieval(rows, index, keyword, fail, other):
    neighbors = NearestNeighbors(n_neighbors=5, metric="cosine").fit(rows[index])
    def scores(query):
        ids = neighbors.kneighbors(rows[query], return_distance=False)
        return keyword[index[ids]].mean(axis=1)

    pos, neg = scores(fail), scores(other)
    return {"failure_fraction_on_keyword": float(pos.mean()),
            "failure_fraction_on_other": float(neg.mean()), "auroc": auroc(pos, neg)}


def coverage(rows, index, keyword, id_queries, ood_queries):
    norm = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    train = norm[index]
    query = norm[np.r_[id_queries, ood_queries]]
    memory = BidirectionalExperienceMemory(dim=rows.shape[1], kde_bandwidth=0.3)
    for i in index:
        memory.add_experience(rows[i], -1.0 if keyword[i] else 1.0, str(i))
    scores = {
        "max_cosine": (query @ train.T).max(axis=1),
        "euclidean_log_kde_bw_0.5": KernelDensity(bandwidth=0.5).fit(train).score_samples(query),
        "bem_coverage_bw_0.3": [memory.coverage_signal(q) for q in rows[np.r_[id_queries, ood_queries]]],
    }
    split = len(id_queries)
    return {name: auroc(values[:split], values[split:]) for name, values in scores.items()}


def newsgroups(encoder, offline):
    dataset = fetch_20newsgroups(subset="all", categories=[*SCIENCE, *SPORTS],
                                 remove=("headers", "footers", "quotes"),
                                 shuffle=True, random_state=42, download_if_missing=not offline)
    texts = dataset.data
    category = np.array(dataset.target_names)[dataset.target]
    science = np.isin(category, SCIENCE)
    keyword = np.array(["orbit" in s.lower() or "circuit" in s.lower() for s in texts])
    rng = np.random.RandomState(42)
    train_fail, test_fail = train_test_split(np.flatnonzero(science & keyword), test_size=0.2, random_state=rng)
    train_other, test_other = train_test_split(np.flatnonzero(science & ~keyword), test_size=0.2, random_state=rng)
    balanced_other = train_other[:len(train_fail)]
    balanced, all_train = np.r_[train_fail, balanced_other], np.r_[train_fail, train_other]
    all_test = [test_fail, test_other, np.flatnonzero(~science)]
    clean_test = unseen_texts(texts, all_train, all_test)
    raw = encoder.encode(texts, device="cpu", batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    projected = keyword_projection(raw, train_fail, balanced_other)
    digest = hashlib.sha256()
    for cat, body in zip(category, texts):
        b = body.encode("utf-8")
        digest.update(cat.encode("ascii") + b"\0" + len(b).to_bytes(8, "big") + b)
    result = {
        "dataset": "20 Newsgroups by date, sklearn.datasets.fetch_20newsgroups(subset='all')",
        "categories_id": list(SCIENCE), "categories_ood": list(SPORTS), "corpus_sha256": digest.hexdigest(),
        "documents": len(texts), "split_seed": 42, "projection_seed": 42,
        "train_keyword": len(train_fail), "train_other": len(train_other),
        "balanced_index": len(balanced), "full_index": len(all_train),
    }
    for cohort, (fail, other, ood) in [("document_split", all_test), ("unique_unseen_bodies", clean_test)]:
        table = {"keyword_queries": len(fail), "other_science_queries": len(other), "sports_queries": len(ood)}
        for name, rows in [("raw", raw), ("projected", projected)]:
            table[name] = {
                "retrieval_balanced": retrieval(rows, balanced, keyword, fail, other),
                "retrieval_full": retrieval(rows, all_train, keyword, fail, other),
                "coverage_balanced": coverage(rows, balanced, keyword, other, ood),
            }
        result[cohort] = table
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--offline", action="store_true", help="require the cached model and corpus")
    args = parser.parse_args()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        parser.exit(1, f"Install the text extra with python -m pip install -e '.[text]': {exc}\n")
    torch.set_num_threads(4)
    with threadpool_limits(limits=1, user_api="blas"):
        encoder = SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu", local_files_only=args.offline)
        result = {
            "model": MODEL_ID, "revision": REVISION, "device": "cpu",
            "packages": {name: version(name) for name in ("numpy", "scikit-learn", "torch",
                                                          "sentence-transformers", "transformers")},
            "authored_fixture": authored_fixture(encoder), "newsgroups": newsgroups(encoder, args.offline),
        }
    write_json(args.output, result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
