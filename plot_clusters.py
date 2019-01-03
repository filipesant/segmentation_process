import time as time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits.mplot3d as p3
from open3d.linux.open3d import *

from sklearn.svm import SVR


from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans, AffinityPropagation, SpectralClustering, \
    Birch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph
from sklearn import mixture


def dpgmm(cloud, n_clusters):
    dpgmm = mixture.BayesianGaussianMixture(n_components=2,
                                            covariance_type='tied',
                                            weight_concentration_prior=1e+2,
                                            weight_concentration_prior_type='dirichlet_distribution',
                                            # weight_concentration_prior_type='dirichlet_process',
                                            mean_precision_prior=1e-4,
                                            # covariance_prior=1e0 * np.eye(3),
                                            init_params='kmeans',
                                            # max_iter=100,
                                            # random_state=None,
                                            warm_start=True).fit(cloud)
    return dpgmm


def gmm(cloud, n_clusters):
    # Fit a Gaussian mixture with EM using ten components
    gmm = mixture.GaussianMixture(n_components=n_clusters,
                                  covariance_type='full',
                                  max_iter=100).fit(cloud)
    result_label = gmm
    return result_label
    # plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
    #              'Expectation-maximization')


def birch_clustering(cloud, n_clusters):
    brc = Birch(branching_factor=50,
                n_clusters=n_clusters,
                threshold=0.001,
                compute_labels=True).fit(cloud)
    result_label = brc.labels_
    return result_label


def spectral_clustering(cloud, n_clusters):
    spectral = SpectralClustering(n_clusters=n_clusters,
                                  eigen_solver='arpack',
                                  affinity="nearest_neighbors").fit(cloud)
    result_label = spectral.labels_
    return result_label


def affinity_propagation(cloud):
    af = AffinityPropagation(preference=-50).fit(cloud)
    result_label = af.labels_
    return result_label


def k_means(cloud, n_clusters):
    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=2)
    k_means.fit(cloud)
    result_label = k_means.labels_
    return result_label


def mini_batch(cloud, batch_size, n_clusters):
    mbk = MiniBatchKMeans(init='k-means++',
                          n_clusters=n_clusters,
                          batch_size=batch_size,
                          n_init=10,
                          max_no_improvement=10,
                          verbose=0)

    mbk.fit(cloud)
    result_label = mbk.labels_
    return result_label


def db_scan(cloud):
    db = DBSCAN(eps=0.02, min_samples=10).fit(cloud)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    ret_labels = db.labels_
    return ret_labels


def agglomerative_clustering(cloud, n_clusters, m_connectivity=None):
    """

    """
    if m_connectivity is None:
        ward = AgglomerativeClustering(linkage='ward',
                                       n_clusters=n_clusters).fit(cloud)
    else:
        ward = AgglomerativeClustering(n_clusters=n_clusters,
                                       connectivity=m_connectivity,
                                       linkage='ward').fit(cloud)

    result_label = ward.labels_
    return result_label


def svr(cloud):
    y = np.sin(cloud).ravel()
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1).fit(cloud,y)
    # svr_rbf = SVR().fit(cloud)
    result_label = svr_rbf.labels_
    return result_label


def plot_cloud(cloud, result_label, title=None, m_axisx=None, m_axisy=None, m_current=None, fig_plt=None):
    """


    """
    fig_plt = plt.figure(figsize=plt.figaspect(0.5))
    axis_x = fig_plt.add_subplot(m_axisx, m_axisy, m_current, projection='3d')
    # axis_x = p3.Axes3D(ax)
    # axis_x.view_init(1000, 1000)
    for l in np.unique(result_label):
        axis_x.scatter(cloud[result_label == l, 0], cloud[result_label == l, 1], cloud[result_label == l, 2],
                       color=plt.cm.jet(np.float(l) / np.max(result_label + 1)),
                       s=20, edgecolor='k')
    plt.title(title)
    plt.show()


def plot_one_cloud(cloud, result_label, title=None):
    """


    """
    fig_plt = plt.figure(figsize=plt.figaspect(0.5))
    # axis_x = fig_plt.add_subplot(m_axisx, m_axisy, m_current, projection='3d')
    axis_x = fig_plt.add_subplot(111, projection='3d')
    # axis_x = p3.Axes3D(ax)
    axis_x.view_init(7, -80)
    axis_x.grid(False)
    axis_x.axis('off')

    test_x = []
    test_y = []
    test_z = []

    for l in np.unique(result_label):
        test_x.append(cloud[result_label == l, 0])
        test_y.append(cloud[result_label == l, 1])
        test_z.append(cloud[result_label == l, 2])


        axis_x.scatter(cloud[result_label == l, 0],
                              cloud[result_label == l, 1],
                              cloud[result_label == l, 2],
                              color=plt.cm.jet(np.float(l) / np.max(result_label + 1)),
                              s=20,
                              edgecolor='k')

    # X, Y = np.meshgrid(np.array(test_x), np.array(test_y))
    # axis_x.contour(X,Y,np.zeros(len(X),len(Y)))

    plt.title(title)
    plt.show()


if __name__ == "__main__":
    # Reading and Downsampling Point Cloud test_pcd
    # pcd = read_point_cloud("files/file1.pcd")
    # pcd = read_point_cloud("/home/filipe/Projects/pcl_sandbox/SegmentationBenchmark/filipe2_teste.pcd")
    pcd = read_point_cloud("/home/filipe/Projects/pcl_sandbox/Downsample/test55_h_plane_removed.pcd")
    # pcd = read_point_cloud("/home/filipe/Projects/Dataset/OSD/pcd/test0.pcd")
    # pcd = voxel_down_sample(pcd, voxel_size=0.03)


    X = np.asarray(pcd.points)
    cluster_number = 14

    # Compute Agglomerative clustering
    st = time.time()
    label = agglomerative_clustering(X, cluster_number)
    plot_one_cloud(X, label, "Agglomerative clustering")
    elapsed_time = time.time() - st
    print("Elapsed time: %.2fs" % elapsed_time)
    print("Number of points: %i" % label.size)

    # Compute DBSCAN clustering
    st = time.time()
    label = db_scan(X)
    elapsed_time = time.time() - st
    # plot_cloud(X, label, "DBSCAN clustering", 3, 3, 3, fig_plt)
    plot_one_cloud(X, label, "DBSCAN clustering")
    print("Elapsed time: %.2fs" % elapsed_time)
    print("Number of points: %i" % label.size)

    # Compute K-Means clustering
    st = time.time()
    label = k_means(X, cluster_number)
    elapsed_time = time.time() - st
    # plot_cloud(X, label, "K-Means Clustering", 3, 3, 4, fig_plt)
    plot_one_cloud(X, label, "K-Means Clustering")
    print("Elapsed time: %.2fs" % elapsed_time)
    print("Number of points: %i" % label.size)

    # Compute Spectral clustering
    st = time.time()
    label = spectral_clustering(X, cluster_number)
    elapsed_time = time.time() - st
    # plot_cloud(X, label, "Spectral Clustering", 3, 3, 7, fig_plt)
    plot_one_cloud(X, label, "Spectral clustering")
    print("Elapsed time: %.2fs" % elapsed_time)
    print("Number of points: %i" % label.size)


    plt.show()

    / home / filipe / Projects / ClusterComparison