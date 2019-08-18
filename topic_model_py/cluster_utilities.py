
# coding: utf-8

# In[2]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...  utility function for cluster analysis on movie reviews
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

import pandas as pd
import statistics

import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import random
from matplotlib.font_manager import FontProperties    

from sklearn.cluster import KMeans

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...  k-means from sklearn
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def k_means(feature_matrix, num_clusters = 5) :
    
    km = KMeans(n_clusters = num_clusters, max_iter = 10000)
    km.fit(feature_matrix)
    clusters = km.labels_
    
    return km, clusters

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...  organize clustering results into data frame
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def make_cluster_agg(clustering_obj, df_to_cluster, 
                        feature_names, num_clusters,
                        topn_features,
                        movie_list, movie_list_abbr):
    
    df_cluster_agg = pd.DataFrame(columns = ['rating_mean', 'rating_median', 'num_ratings',
#                                             'STP', 'SAP', 'TBL', 'FAR',
                                             'movie_cnt', 'key_features'])

# get cluster centroids    
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
        
# get key features for each cluster
    
    for cluster_num in range(num_clusters):
                
        key_features = [feature_names[index] 
                        for index in ordered_centroids[cluster_num, :topn_features]]
                        
        movie = df_to_cluster[df_to_cluster['cluster'] == cluster_num]['movie_name'].values.tolist()
        rating = df_to_cluster[df_to_cluster['cluster'] == cluster_num]['review_rating'].values.tolist()
        title =  df_to_cluster[df_to_cluster['cluster'] == cluster_num]['review_title'].values.tolist()
        
        df_cluster_agg.at[cluster_num, 'key_features'] = key_features
        
        rating_clean = [float(x) for x in rating if x > 0]
        
        df_cluster_agg.at[cluster_num, 'rating_mean'] = round(sum(rating_clean) / len(rating_clean), 2)
        df_cluster_agg.at[cluster_num, 'rating_median'] = statistics.median(rating_clean)
        
        df_cluster_agg.at[cluster_num, 'movie_cnt'] = [[x, movie.count(x)] for x in set(movie)]
        df_cluster_agg.at[cluster_num, 'num_ratings'] = len(movie)
        
    for k in range(num_clusters) :
        lst = df_cluster_agg['movie_cnt'][[k]]
        for i in range(len(lst[k])) :
            for m in range(len(movie_list)) :
                if (lst[k][i][0] == movie_list[m]) :
                    df_cluster_agg.at[k, movie_list_abbr[m]] = lst[k][i][1]
                        
    del df_cluster_agg['movie_cnt']
    
    df_cluster_agg = df_cluster_agg.fillna(0)
    
    return df_cluster_agg

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...  organize cluster data into summary dictionary
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def get_cluster_data(clustering_obj, df_to_cluster, 
                     feature_names, num_clusters,
                     topn_features = 10):
    
    cluster_details = {}
    
    # get cluster centroids
    
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
        
    # get key features for each cluster
    # get movies belonging to each cluster
    
    for cluster_num in range(num_clusters):
        
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        
        key_features = [feature_names[index] 
                        for index in ordered_centroids[cluster_num, :topn_features]]
        
        cluster_details[cluster_num]['key_features'] = key_features
        
        movie = df_to_cluster[df_to_cluster['cluster'] == cluster_num]['movie_name'].values.tolist()
        rating = df_to_cluster[df_to_cluster['cluster'] == cluster_num]['review_rating'].values.tolist()
        title =  df_to_cluster[df_to_cluster['cluster'] == cluster_num]['review_title'].values.tolist()
        
        cluster_details[cluster_num]['movie'] = movie
        cluster_details[cluster_num]['rating'] = rating
        cluster_details[cluster_num]['title'] = title
    
    return cluster_details     
    
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...  basic text output of cluster data from summary dictionary
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def print_cluster_data(cluster_data):
    # print cluster details
    for cluster_num, cluster_details in cluster_data.items():
        print 'Cluster {} details:'.format(cluster_num)
        print '-'*20
        
        print 'Key features:', cluster_details['key_features']
        
        print list(collections.Counter(cluster_details['movie']).items())
                
        clean_list = [float(x) for x in cluster_details['rating'] if x > 0]
        
        print 'Median Rating : '
        print statistics.median(clean_list)
        
        print 'Mean Rating : '
        cluster_mean = sum(clean_list) / len(clean_list)
        print cluster_mean
        
        print '='*40
        

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...  plot movie titles on 2D map, grouped by cluster features
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def plot_clusters(num_clusters, feature_matrix,
                  cluster_data, movie_data,
                  plot_size=(16,8)):
    
    # generate random color for clusters                  
    def generate_random_color():
        color = '#%06x' % random.randint(0, 0xFFFFFF)
        return color
    
# ... http://colorbrewer2.org/#type=qualitative&scheme=Set1&n=6
    marker_color = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
        
    # define markers for clusters    
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
        
    # build cosine distance matrix
    cosine_distance = 1 - cosine_similarity(feature_matrix) 
        
    # dimensionality reduction using MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", 
              random_state=1)
        
    # get coordinates of clusters in new low-dimensional space
    plot_positions = mds.fit_transform(cosine_distance)  
    x_pos, y_pos = plot_positions[:, 0], plot_positions[:, 1]
        
    # build cluster plotting data
    cluster_color_map = {}
    cluster_name_map = {}
        
    for cluster_num, cluster_details in cluster_data.items():
        
        # assign cluster features to unique label
#        cluster_color_map[cluster_num] = generate_random_color()
        cluster_color_map[cluster_num] = marker_color[cluster_num]
        cluster_name_map[cluster_num] = ', '.join(cluster_details['key_features'][:5]).strip()
        
    # map each unique cluster label with its coordinates and movies
    cluster_plot_frame = pd.DataFrame({'x': x_pos,
                                       'y': y_pos,
                                       'label': movie_data['cluster'].values.tolist(),
                                       'title': movie_data['movie_name'].values.tolist()
                                        })
    grouped_plot_frame = cluster_plot_frame.groupby('label')
        
    # set plot figure size and axes
    fig, ax = plt.subplots(figsize=plot_size) 
    ax.margins(0.05)
        
    # plot each cluster using co-ordinates and movie titles
    for cluster_num, cluster_frame in grouped_plot_frame:
         marker = markers[cluster_num] if cluster_num < len(markers)                   else np.random.choice(markers, size=1)[0]
         ax.plot(cluster_frame['x'], cluster_frame['y'], 
                 marker=marker, linestyle='', ms=12,
                 label=cluster_name_map[cluster_num], 
                 color=cluster_color_map[cluster_num], mec='none')
         ax.set_aspect('auto')
         ax.tick_params(axis= 'x', which='both', bottom='off', top='off',        
                        labelbottom='off')
         ax.tick_params(axis= 'y', which='both', left='off', top='off',         
                        labelleft='off')
        
    fontP = FontProperties()
    fontP.set_size('small')    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), fancybox=True, 
              shadow=True, ncol=5, numpoints=1, prop=fontP) 
        
    #add labels as the film titles
    for index in range(len(cluster_plot_frame)):
        ax.text(cluster_plot_frame.ix[index]['x'], 
                cluster_plot_frame.ix[index]['y'], 
                cluster_plot_frame.ix[index]['title'], size=8)  
        
    # show the plot           
    plt.show() 
    

