import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
import glob
import numpy as np

def kmeans_score(x, y, weights, k):
    from sklearn.cluster import KMeans
    from collections import Counter
    
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    cluster = KMeans(n_clusters=k)
    cluster.fit(x, sample_weight=weights)
    distances = np.zeros(y.shape)
    for i in range(k):
        inds_per_cluster = (cluster.labels_==i)
        # count number of gt per cluster
        unique, counts = np.unique(y[inds_per_cluster], return_counts=True)
        # efficiency mapping gt to covering
        mapping_covering = np.zeros(unique.max()+1,dtype=counts.dtype)
        mapping_covering[unique] = counts
        covering_score = 1.0 - mapping_covering[y[inds_per_cluster]]*1.0/counts.sum()
        # calculate distance of sample to its centroid from L2 distance -> Cosine distance
        distance_to_centroids = cluster.transform(x[inds_per_cluster]) / 2
        distance_to_its_centroid = np.array([distance_to_centroids[i,j] for i, j in enumerate(cluster.labels_[inds_per_cluster])])
        max_distance_to_centroids = distance_to_centroids.max(axis=-1)
        distances[inds_per_cluster] = (covering_score + (1-(max_distance_to_centroids - distance_to_its_centroid))) / 2

    return distances

def knn_score(x, y, indexes, k):
    from sklearn.neighbors import KNeighborsClassifier
    def distance(x, y, **kwargs):
        return (((x-y)**2).sum())**0.5
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    model = KNeighborsClassifier(# metric=distance, p=2,
                                metric='minkowski', p=2,
                                # metric_params=dict(w=weights)
                                )
    model.fit(x[indexes], y[indexes])
    idxs = model.kneighbors(x, k, False)
    distances = np.zeros(x.shape[0])
    for i, idx in enumerate(idxs):
        covering_score = 1.0 - (y[idx]==y[i]).sum() / k
        distance_score = ((x[i] - x[idx])**2).sum()**0.5 / 2.0
        distances[i] = covering_score * distance_score
        assert covering_score <= 1.0 and distance_score <= 1.0
    return distances

if __name__=="__main__":
    from sklearn.cluster import KMeans
    from collections import Counter
    k = 4
    n = 20
    x = np.random.rand(n, 5)
    y = np.random.randint(0, k, n)
    idx = np.random.randint(0, n, n//3)
    print(knn_score(x, y, idx, k))
    # idx = np.array([x[i,j] for i, j in enumerate(y)])
    # print(x, y, idx)
    # print((np.random.rand(n, 1) * np.random.rand(n, 1)).shape)
        
    # paths = glob.glob('debug/SIRC_noise03/*.pickle', recursive=True)
    # colors = np.random.randint(0, 10, (10, 3))/10
    # print(colors)
    # for path in paths:
    #     print(path)
    #     with open(path, 'rb') as f:
    #         data = pickle.load(f)
    #     labels, logits, features, idxs = data['labels'].squeeze().cpu().detach().numpy().tolist(), \
    #                                     data['logits'].cpu().detach().numpy(), data['features'].cpu().detach().numpy(),\
    #                                     data['idxs']
    #     tsne = TSNE(n_components=2, random_state=10)
    #     embedded_data = tsne.fit_transform(features)

    #     plt.figure(figsize=(10, 8))
    #     for label in set(labels):
    #         indices = [i for i, l in enumerate(labels) if l == label]
    #         indices_highlight = list(set(indices).intersection(set(idxs)))
    #         indices_normal = list(set(indices) - set(indices_highlight))
    #         plt.scatter(embedded_data[indices_highlight, 0], embedded_data[indices_highlight, 1], label=label, c=colors[label].reshape(1, -1), marker='x')
    #         plt.scatter(embedded_data[indices_normal, 0], embedded_data[indices_normal, 1], label=label, c=colors[label].reshape(1, -1), alpha=0.4)
    #     plt.legend()
    #     # plt.show()
    #     plt.savefig(path.replace('.pickle', '.png'))