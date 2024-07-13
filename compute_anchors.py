import argparse
import json
import yaml
import random
import numpy as np
from sklearn.cluster import KMeans

SEED = 42
NUM_CLUSTERS = 9
CONFIG_PATH = "config/config.yaml"
random.seed(SEED)
np.random.seed(SEED)

def get_json_data(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    f.close()
    return data

def set_config_anchors(sm_anchors: np.ndarray, md_anchors: np.ndarray, lg_anchors: np.ndarray):
    with open(CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f)
        data["anchors"]["sm"] = sm_anchors.tolist()
        data["anchors"]["md"] = md_anchors.tolist()
        data["anchors"]["lg"] = lg_anchors.tolist()
    f.close()
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(data, f)
    f.close()

if __name__ == "__main__":
    annotations_path = "dataset/annotations/MD_mapping.json"
    annotator = "annotator_a"
    init = "k-means++"
    n_init = "auto"
    max_iter = 500
    tol = 1e-10

    parser = argparse.ArgumentParser(description=f"Anchor Segment Generation")
    parser.add_argument(
        "--annotations_path", type=str, default=annotations_path, metavar="", 
        help=f"JSON annotations path (default = {annotations_path})"
    )
    parser.add_argument(
        "--annotator", type=str, default=annotator, metavar="", 
        help=f"Specific annotator key (if multiple, else use 'annotator_a') (default = {annotator})"
    )
    parser.add_argument(
        "--init", type=str, default=init, metavar="", choices=["k-means++", "random"], 
        help=f"Cluster initialisation technique (k-means++ or random) (default = {init})"
    )
    parser.add_argument(
        "--n_init", type=str, default=n_init, metavar="", 
        help=f"Number of times the cluster algorithm is run with different centroid seeds (default = {n_init})"
    )
    parser.add_argument(
        "--max_iter", type=int, default=max_iter, metavar="", 
        help=f"Number of clustering iterations (default = {max_iter})"
    )
    parser.add_argument(
        "--tol", type=int, default=tol, metavar="", 
        help=f"Tolerance of clustering algorithm (default = {tol})"
    )
    args = parser.parse_args()

    annotations = get_json_data(args.annotations_path)
    annotations = annotations["annotations"][args.annotator]

    durations = [i["end"]-i["start"] for segments in list(annotations.values()) for i in segments.values()]
    durations = np.asarray(durations).reshape(-1, 1)
    cluster_model = KMeans(
        NUM_CLUSTERS, 
        init=args.init, 
        n_init=args.n_init if (not args.n_init.isnumeric()) else int(args.n_init), 
        tol=args.tol, 
        max_iter=args.max_iter
    )
    cluster_model.fit(durations)

    anchors = cluster_model.cluster_centers_.reshape(-1)
    anchors = anchors[np.argsort(anchors)]
    sm_anchors, md_anchors, lg_anchors = anchors[:3], anchors[3:6], anchors[6:]
    set_config_anchors(sm_anchors, md_anchors, lg_anchors)
    