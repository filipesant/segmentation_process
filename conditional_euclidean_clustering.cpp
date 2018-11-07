
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/time.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <vector>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include <thread>
#include <sstream>
#include <boost/format.hpp>
#include <pcl/segmentation/region_growing_rgb.h>

#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>

#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/extract_clusters.h>

#include "kmeans.cpp"

typedef pcl::PointXYZRGB PointTypeIO;
typedef pcl::PointXYZRGBNormal PointTypeFull;

using namespace std;
using namespace pcl;

bool enforceIntensitySimilarity(const PointTypeFull &point_a, const PointTypeFull &point_b, float squared_distance)
{
  return (false);
}

bool enforceCurvatureOrIntensitySimilarity(const PointTypeFull &point_a, const PointTypeFull &point_b, float squared_distance)
{
  return (false);
}

bool customRegionGrowing(const PointTypeFull &point_a, const PointTypeFull &point_b, float squared_distance)
{
  Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap();
  Eigen::Map<const Eigen::Vector3f> point_b_normal = point_b.getNormalVector3fMap();

  // std::cout << "squared_distance : " << squared_distance << std::endl;
  // // std::cout << "intesity : " << fabs(point_a.intensity - point_b.intensity) << std::endl;
  // std::cout << "dot product : " << fabs(point_a_normal.dot(point_b_normal)) << std::endl;
  // std::cout << " " << std::endl;

  if ((squared_distance < 0.00005))
  {
    return true;
  }
}

void showPointCloud(PointCloud<PointTypeIO>::Ptr cloud, int index, bool colorfy)
{
  boost::shared_ptr<visualization::PCLVisualizer> viewer(new visualization::PCLVisualizer("3D Viewer"));
  GlasbeyLUT colors;

  stringstream cluster_id;  
  cluster_id << "cluster_" << index;

  visualization::PointCloudColorHandlerCustom<PointTypeIO> rgb(cloud, colors.at(index).r, colors.at(index).g, colors.at(index).b);
  viewer->setBackgroundColor(0, 0, 0);

  if(colorfy){
    viewer->addPointCloud<PointTypeIO>(cloud, rgb, cluster_id.str());
  } else {
    viewer->addPointCloud<PointTypeIO>(cloud, cluster_id.str());
  }

  // viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 10, cluster_id.str());
  // viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_OPACITY, 1, cluster_id.str());
  viewer->setBackgroundColor(1, 1, 1);
  viewer->setCameraPosition(0.0232564, 0.061806, -0.71835, -0.0038714, -0.995991, -0.0893641);

  while (!viewer->wasStopped())
  {
    viewer->spinOnce(100);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }

  return;
}

void showOneByOne(std::vector<PointCloud<PointTypeIO>::Ptr> clusters_list, bool colorfy)
{
  for (vector<PointCloud<PointTypeIO>::Ptr>::iterator it = clusters_list.begin(); it != clusters_list.end(); ++it)
  {
    showPointCloud(*it,it - clusters_list.begin(),colorfy);
  }

}

void colorfyClusters(std::vector<PointCloud<PointTypeIO>::Ptr> clusters_list, int pointSize , bool colorfy = true)
{
  boost::shared_ptr<visualization::PCLVisualizer> viewer(new visualization::PCLVisualizer("3D Viewer"));
  GlasbeyLUT colors;

  int index = 0;

  for (vector<PointCloud<PointTypeIO>::Ptr>::iterator it = clusters_list.begin(); it != clusters_list.end(); ++it)
  {
    stringstream cluster_id;
    index = it - clusters_list.begin();
    cluster_id << "cluster_" << index;

    visualization::PointCloudColorHandlerCustom<PointTypeIO> rgb(*it, colors.at(index).r, colors.at(index).g, colors.at(index).b);
    viewer->setBackgroundColor(0, 0, 0);

    if(colorfy){
      viewer->addPointCloud<PointTypeIO>(*it, rgb, cluster_id.str());
    } else {
      viewer->addPointCloud<PointTypeIO>(*it, cluster_id.str());
    }
    
    viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, pointSize, cluster_id.str());
    viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_OPACITY, 1, cluster_id.str());
    viewer->setBackgroundColor(1, 1, 1);
    viewer->setCameraPosition(0.0232564, 0.061806, -0.71835, -0.0038714, -0.995991, -0.0893641);
  }

  while (!viewer->wasStopped())
  {
    viewer->spinOnce(100);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }

  return;
}

void regionGrowingColor(vector<PointCloud<PointTypeIO>::Ptr> cloud_list,
                        float distanceThreshold,
                        float pointColorThreshold,
                        float regionColorThreshold,
                        int minClusterSizeRG,
                        int pointSize)
{
  cout << "------------Region Growing Color------------- " << endl;
  cout << "distanceThreshold " << distanceThreshold << endl;
  cout << "pointColorThreshold " << pointColorThreshold << endl;
  cout << "regionColorThreshold " << regionColorThreshold << endl;
  cout << "minClusterSizeRG " << minClusterSizeRG << endl;

  pcl::search::Search<PointTypeIO>::Ptr tree = boost::shared_ptr<pcl::search::Search<PointTypeIO>>(new pcl::search::KdTree<PointTypeIO>);

  int index = 0;

  vector<PointCloud<PointTypeIO>::Ptr> rg_result;

  for (vector<PointCloud<PointTypeIO>::Ptr>::iterator it = cloud_list.begin(); it != cloud_list.end(); ++it)
  {

    pcl::RegionGrowingRGB<PointTypeIO> reg;
    reg.setInputCloud(*it);
    // reg.setIndices (indices);
    reg.setSearchMethod(tree);
    reg.setDistanceThreshold(distanceThreshold);
    reg.setPointColorThreshold(pointColorThreshold);
    reg.setRegionColorThreshold(regionColorThreshold);
    reg.setMinClusterSize(minClusterSizeRG);

    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);

    vector<PointCloud<PointTypeIO>::Ptr> rg_clusters_result;

    int j = 0;
    for (int i = 0; i < clusters.size(); ++i)
    {
      PointCloud<PointTypeIO>::Ptr cloud_cluster(new PointCloud<PointTypeIO>);
      for (vector<int>::const_iterator pit = (clusters)[i].indices.begin(); pit != (clusters)[i].indices.end(); ++pit)
      {
        cloud_cluster->points.push_back((*it)->points[*pit]);
      }
      cloud_cluster->width = cloud_cluster->points.size();
      cloud_cluster->height = 1;
      cloud_cluster->is_dense = true;
      rg_clusters_result.push_back(cloud_cluster);
      rg_result.push_back(cloud_cluster);
      j++;
    }

    cout << "SIZE : " << rg_clusters_result.size() << endl;

    // colorfyClusters(rg_clusters_result);
  }

  colorfyClusters(rg_result, pointSize);
}

vector<PointCloud<PointTypeIO>::Ptr> euclideanCLustering(PointCloud<PointTypeIO>::Ptr cloud, float clusterTolerance, int minClusterSize, int maxClusterSize, float radiusSearch)
{
  cout << "clusterTolerance " << clusterTolerance << endl;
  cout << "minClusterSize " << minClusterSize << endl;
  cout << "maxClusterSize " << maxClusterSize << endl;
  cout << "radiusSearch " << radiusSearch << endl;

  // Data containers used
  pcl::PointCloud<PointTypeIO>::Ptr cloud_out(new pcl::PointCloud<PointTypeIO>);
  pcl::PointCloud<PointTypeFull>::Ptr cloud_with_normals(new pcl::PointCloud<PointTypeFull>);
  pcl::IndicesClustersPtr clusters(new pcl::IndicesClusters), small_clusters(new pcl::IndicesClusters), large_clusters(new pcl::IndicesClusters);
  pcl::search::KdTree<PointTypeIO>::Ptr search_tree(new pcl::search::KdTree<PointTypeIO>);
  pcl::console::TicToc tt;

  copyPointCloud(*cloud, *cloud_out);

  // Set up a Normal Estimation class and merge data in cloud_with_normals
  pcl::copyPointCloud(*cloud_out, *cloud_with_normals);

  pcl::NormalEstimation<PointTypeIO, PointTypeFull> ne;
  ne.setInputCloud(cloud_out);
  ne.setSearchMethod(search_tree);
  ne.setRadiusSearch(radiusSearch);
  ne.compute(*cloud_with_normals);

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<PointTypeIO>::Ptr tree(new pcl::search::KdTree<PointTypeIO>);
  tree->setInputCloud(cloud_out);

  pcl::EuclideanClusterExtraction<PointTypeIO> ec;
  ec.setClusterTolerance(clusterTolerance); // 2cm
  ec.setMinClusterSize(minClusterSize);
  ec.setMaxClusterSize(maxClusterSize);
  ec.setSearchMethod(tree);
  // ec.setIndices(ne);
  ec.setInputCloud(cloud_out);
  ec.extract(*clusters);

  // Set up a Conditional Euclidean Clustering class
  // pcl::ConditionalEuclideanClustering<PointTypeFull> cec(true);
  // cec.setInputCloud(cloud_with_normals);
  // cec.setConditionFunction(&customRegionGrowing);
  // cec.setClusterTolerance(clusterTolerance);
  // cec.setMinClusterSize(minClusterSize);
  // cec.setMaxClusterSize(maxClusterSize);

  // cec.segment(*clusters);
  // cec.getRemovedClusters(small_clusters, large_clusters);

  vector<PointCloud<PointTypeIO>::Ptr> euclidean_clusters_result;

  int j = 0;
  for (int i = 0; i < clusters->size(); ++i)
  {
    PointCloud<PointTypeIO>::Ptr cloud_cluster(new PointCloud<PointTypeIO>);
    for (vector<int>::const_iterator pit = (*clusters)[i].indices.begin(); pit != (*clusters)[i].indices.end(); ++pit)
    {
      cloud_cluster->points.push_back(cloud_out->points[*pit]);
    }
    cloud_cluster->width = cloud_cluster->points.size();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;
    euclidean_clusters_result.push_back(cloud_cluster);
    j++;
  }

  return euclidean_clusters_result;
}

// void regionGrowing(vector<PointCloud<PointTypeIO>::Ptr> cloud_list)
void regionGrowing(vector<PointCloud<PointTypeIO>::Ptr> cloud_list, int kSearch, int minClusterSize, int maxClusterSize, int numberOfNeighbours, float smoothnessThreshold, float curvatureThreshold, int pointSize)
{
  vector<PointCloud<PointTypeIO>::Ptr> rg_result;

  for (vector<PointCloud<PointTypeIO>::Ptr>::iterator it = cloud_list.begin(); it != cloud_list.end(); ++it)
  {

    cout << "kSearch " << kSearch << endl;
    cout << "minClusterSize " << minClusterSize << endl;
    cout << "maxClusterSize " << maxClusterSize << endl;
    cout << "numberOfNeighbours " << numberOfNeighbours << endl;
    cout << "smoothnessThreshold " << smoothnessThreshold << endl;
    cout << "curvatureThreshold " << curvatureThreshold << endl;

    pcl::search::Search<PointTypeIO>::Ptr tree = boost::shared_ptr<pcl::search::Search<PointTypeIO>>(new pcl::search::KdTree<PointTypeIO>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<PointTypeIO, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(*it);
    normal_estimator.setKSearch(kSearch);
    normal_estimator.compute(*normals);

    pcl::RegionGrowing<PointTypeIO, pcl::Normal> reg;
    reg.setMinClusterSize(minClusterSize);
    reg.setMaxClusterSize(maxClusterSize);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(numberOfNeighbours);
    reg.setInputCloud(*it);
    //reg.setIndices (indices);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(smoothnessThreshold); // 3.0 / 180.0 * M_PI
    reg.setCurvatureThreshold(curvatureThreshold);

    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);

    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
    // pcl::visualization::CloudViewer viewer("Cluster viewer");
    // viewer.showCloud(colored_cloud);
    // while (!viewer.wasStopped())
    // { 
    //   boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    // }

    
    vector<PointCloud<PointTypeIO>::Ptr> euclidean_clusters_result;

    int j = 0;
    for (int i = 0; i < clusters.size(); ++i)
    {
      PointCloud<PointTypeIO>::Ptr cloud_cluster(new PointCloud<PointTypeIO>);
      for (vector<int>::const_iterator pit = (clusters)[i].indices.begin(); pit != (clusters)[i].indices.end(); ++pit)
      {
        cloud_cluster->points.push_back((*it)->points[*pit]);
      }
      cloud_cluster->width = cloud_cluster->points.size();
      cloud_cluster->height = 1;
      cloud_cluster->is_dense = true;
      euclidean_clusters_result.push_back(cloud_cluster);
      rg_result.push_back(cloud_cluster);
      j++;
    }
  }

  colorfyClusters(rg_result, pointSize);
 

}

pcl::PointCloud<PointTypeIO>::Ptr kmeans_reduction(PointCloud<PointTypeIO>::Ptr cloud, int maxIterations, int number_k = 2)
{
  // io::savePCDFile("teste_input.pcd", *cloud);
  pcl::PointCloud<PointTypeIO>::Ptr tempCloud(new pcl::PointCloud<PointTypeIO>);
  copyPointCloud(*cloud, *tempCloud);
  // srand(time(NULL));

  int total_points = tempCloud->points.size();
  int total_values = 3;
  int K = number_k;
  int max_iterations = maxIterations;
  int has_name = 0;

  vector<Point> points;
  string point_name;

  for (int i = 0; i < total_points; i++)
  {
    vector<double> values;
    uint32_t rgb = *reinterpret_cast<int *>(&cloud->points[i].rgb);
    uint8_t r = (rgb >> 16) & 0x0000ff;
    uint8_t g = (rgb >> 8) & 0x0000ff;
    uint8_t b = (rgb)&0x0000ff;

    // cout << "RED_" << to_string(i)  << " : " << to_string(r) << endl;

    values.push_back(static_cast<double>(r)); 
    values.push_back(static_cast<double>(g));
    values.push_back(static_cast<double>(b)); 

    // values.push_back(cloud->points[i].r);
    // values.push_back(cloud->points[i].g);
    // values.push_back(cloud->points[i].b);
    // values.push_back(cloud->points[i].a);

    if (has_name)
    {
      cin >> point_name;
      Point p(i, values, point_name);
      points.push_back(p);
    }
    else
    {
      Point p(i, values);
      points.push_back(p);
    }
  }

  KMeans kmeans(K, total_points, total_values, max_iterations);
  kmeans.run(points);
  
  // vector<int> clusters;
  // vector<int> points_clusters;

  // int cluster1_r;
  // int cluster1_g;
  // int cluster1_b;
  // int cluster1_a;

  // int cluster2_r;
  // int cluster2_g;
  // int cluster2_b;
  // int cluster2_a;

  // for (int i = 0; i < points.size(); i++)
  // {
  //   clusters.push_back(points[i].getCluster());
  // }

  // for (int i = 0; i < K; i++)
  // {
  //   int mycount = std::count(clusters.begin(), clusters.end(), i);
  //   points_clusters.push_back(mycount);
  // }

  // for (int i = 0; i < tempCloud->points.size(); i++)
  // {
  //   if (points[i].getCluster() == 0)
  //   {
  //     cluster1_r += tempCloud->points[i].r;
  //     cluster1_g += tempCloud->points[i].g;
  //     cluster1_b += tempCloud->points[i].b;
  //     cluster1_a += tempCloud->points[i].a;
  //   }

  //   if (points[i].getCluster() == 1)
  //   {
  //     cluster2_r += tempCloud->points[i].r;
  //     cluster2_g += tempCloud->points[i].g;
  //     cluster2_b += tempCloud->points[i].b;
  //     cluster2_a += tempCloud->points[i].a;
  //   }
  // }

  // vector<int> rgba1;
  // int c1_r = cluster1_r / points_clusters[0];
  // int c1_g = cluster1_g / points_clusters[0];
  // int c1_b = cluster1_b / points_clusters[0];
  // int c1_a = cluster1_a / points_clusters[0];

  // rgba1.push_back(c1_r);
  // rgba1.push_back(c1_g);
  // rgba1.push_back(c1_b);
  // rgba1.push_back(c1_a);

  // vector<int> rgba2;
  // int c2_r = cluster2_r / points_clusters[1];
  // int c2_g = cluster2_g / points_clusters[1];
  // int c2_b = cluster2_b / points_clusters[1];
  // int c2_a = cluster2_a / points_clusters[1];

  // rgba2.push_back(c2_r);
  // rgba2.push_back(c2_g);
  // rgba2.push_back(c2_b);
  // rgba2.push_back(c2_a);

  // vector<vector<int>> vec;
  // vec.push_back(rgba1);
  // vec.push_back(rgba2);

  
  // vec.push_back(rgba1);
  // vec.push_back(rgba2);
  // clusters[i].getCentralValue(j)
  vector<Cluster> vec = kmeans.getResultClusters();
  for (int i = 0; i < tempCloud->points.size(); i++)
  {
    tempCloud->points[i].r = vec[points[i].getCluster()].getCentralValue(0);
    tempCloud->points[i].g = vec[points[i].getCluster()].getCentralValue(1);
    tempCloud->points[i].b = vec[points[i].getCluster()].getCentralValue(2);
    // tempCloud->points[i].a = vec[points[i].getCluster()].getCentralValue(0);
  }

  // io::savePCDFile("teste.pcd", *tempCloud);

  return tempCloud;
}

vector<PointCloud<PointTypeIO>::Ptr> processKmeans(vector<PointCloud<PointTypeIO>::Ptr> cloud_list, int maxIterations , int number_k)
{

  vector<PointCloud<PointTypeIO>::Ptr> clusters_result;

  for (vector<PointCloud<PointTypeIO>::Ptr>::iterator it = cloud_list.begin(); it != cloud_list.end(); ++it)
  {
    string namePcd = "teste_new_" + std::to_string(it - cloud_list.begin()) + ".pcd";
    io::savePCDFile(namePcd, *kmeans_reduction(*it, maxIterations, number_k));
    clusters_result.push_back(kmeans_reduction(*it, maxIterations, number_k));
  }

  return clusters_result;
}

int main(int argc, char **argv)
{

  vector<string> input;
  ifstream inFile;
  string strFileName = argv[1];
  string pcdPath = argv[2];

  std::ifstream file(strFileName);
  if (file.is_open())
  {
    std::string line;
    while (getline(file, line))
    {
      input.push_back(line);
    }
    file.close();
  }

  //Region Growing Parameters
  float distanceThreshold = atof(input[9].c_str());
  float pointColorThreshold = atof(input[11].c_str());
  float regionColorThreshold = atof(input[13].c_str());
  int minClusterSizeRGC = atoi(input[15].c_str());

  //Euclidean Clustering Parameters
  float clusterTolerance = atof(input[18].c_str());
  int minClusterSize = atoi(input[20].c_str());
  int maxClusterSize = atoi(input[22].c_str());
  float radiusSearch = atof(input[24].c_str());

  //Region Growing Parameters
  int kSearch = atoi(input[27].c_str());
  int minClusterSizeRG = atoi(input[29].c_str());
  int maxClusterSizeRG = atoi(input[31].c_str());
  int numberOfNeighbours = atoi(input[33].c_str());
  float smoothnessThreshold = atof(input[35].c_str());
  float curvatureThreshold = atof(input[37].c_str());

  //Kmeans Parameters
  int maxIterations = atoi(input[40].c_str());
  int numberK = atoi(input[42].c_str());

  //Colorfy Params
  int point_size = atoi(input[45].c_str());

  pcl::PointCloud<PointTypeIO>::Ptr cloud_in(new pcl::PointCloud<PointTypeIO>);
  pcl::io::loadPCDFile(pcdPath, *cloud_in);

  vector<PointCloud<PointTypeIO>::Ptr> euclidean_clusters_result = euclideanCLustering(cloud_in, clusterTolerance, minClusterSize, maxClusterSize, radiusSearch);
  // vector<PointCloud<PointTypeIO>::Ptr> kmeans_clusters_result = processKmeans(euclidean_clusters_result, maxIterations, numberK);

  // showOneByOne(kmeans_clusters_result,false);
  // colorfyClusters(euclidean_clusters_result,false);
  // colorfyClusters(kmeans_clusters_result);
  // regionGrowingColor(kmeans_clusters_result, distanceThreshold, pointColorThreshold, regionColorThreshold, minClusterSizeRGC, point_size);
  regionGrowing(euclidean_clusters_result, kSearch, minClusterSizeRG, maxClusterSizeRG, numberOfNeighbours, smoothnessThreshold, curvatureThreshold, point_size);

  return (0);
}
