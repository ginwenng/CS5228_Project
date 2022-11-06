from category_encoders.ordinal import OrdinalEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from scipy.spatial.distance import pdist

def compute_average_distance(X):
    """
    Computes the average distance among a set of n points in the d-dimensional space.

    Arguments:
        X {numpy array} - the query points in an array of shape (n,d),
                          where n is the number of points and d is the dimension.
    Returns:
        {float} - the average distance among the points
    """
    return np.mean(pdist(X))


class DBScanKNNRecommender:

    """
    Use DBScan algorithm to find cluster of input listing.
    If the listing belongs to the outlier cluster, use KNN algorithm to find top k recommendations.
    Else, select k recommendations from cluster.

    """
    
    def __init__(self):
        self.feature_columns = ['property_type', 'tenure', 'num_beds', 'num_baths',
               'size_sqft', 'price', 'subzone_area_size',
               'subzone_population', 'planning_area_area_size',
               'planning_area_population', 'planning_area_population_density',
               'subzone_population_density', 'subzone_num_pri_sch',
               'planning_area_num_pri_sch', 'subzone_num_sec_sch',
               'planning_area_num_sec_sch', 'subzone_num_malls',
               'planning_area_num_malls', 'subzone_num_comm', 'planning_area_num_comm',
               'subzone_num_BN', 'subzone_num_CR', 'subzone_num_IEBP',
               'subzone_num_IHL', 'planning_area_num_BN', 'planning_area_num_CR',
               'planning_area_num_IEBP', 'planning_area_num_IHL', 'subzone_num_mrt',
               'planning_area_num_mrt', 'subzone_num_cc', 'subzone_num_ce',
               'subzone_num_cg', 'subzone_num_dt', 'subzone_num_ew', 'subzone_num_ne',
               'subzone_num_ns', 'subzone_num_te', 'planning_area_num_cc',
               'planning_area_num_ce', 'planning_area_num_cg', 'planning_area_num_dt',
               'planning_area_num_ew', 'planning_area_num_ne', 'planning_area_num_ns',
               'planning_area_num_te']

        self.show_columns = ['listing_id', 'title', 'address', 'property_name', 'property_type',
       'tenure', 'num_beds', 'num_baths', 'size_sqft', 'property_details_url',
       'lat', 'lng', 'planning_area', 'subzone', 'price']
    
    def fit(self, X):

        """
        Transforms input dataframe into numerical feature columns.

        """
        
        self.all_columns = X.columns.tolist()
        
        self.all_X = X
        
        X = X[self.feature_columns]
        
        self.enc = OrdinalEncoder(
            cols=['tenure', 'property_type'],
            mapping=[
                {'col': 'tenure', 'mapping': {'99-year leasehold': 0, '999-year leasehold': 1}},
                {'col': 'property_type', 'mapping': {
                    'hdb': 0, 
                    'walk-up': 1,
                    'apartment': 1,
                    'executive condo': 1,
                    'condo': 2,
                    'townhouse': 2,
                    'shophouse': 2,
                    'cluster house': 2,
                    'landed': 2,
                    'terraced house': 2,
                    'corner terrace': 2,
                    'semi-detached house': 2,
                    'bungalow': 2,
                    'good class bungalow': 2,
                    'land only': 2,
                }},
            ]
        )
        X[['tenure', 'property_type']] = self.enc.fit_transform(X[['tenure', 'property_type']])
        
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        self.X_to_compare = X
        
        return self

    def get_overall_alignment_score(self, full_df):

        """
        Get top 10 recommendations for each row in input dataframe and calculate alignment score and
        average distance among target + recommendation points.

        """
        
        num_records = len(full_df)
        total_scores = 0
        total_distance = 0

        # fit DBScan model
        dbscan = DBSCAN(eps=0.3, min_samples=11)
        dbscan.fit(self.X_to_compare)

        # fit KNN model
        neigh = NearestNeighbors(n_neighbors=11)
        neigh.fit(self.X_to_compare)

        for i in range(num_records):
            score = 0
            original_X = full_df.iloc[i]

            # get DBScan cluster of current row
            target_cluster = dbscan.labels_[i]

            # if in outlier cluster, get recommendations from KNN
            if target_cluster==-1:
                X = pd.DataFrame(
                    data=[original_X.values],
                    columns=self.all_columns
                )

                X = X[self.feature_columns]

                X[['tenure', 'property_type']] = self.enc.transform(X[['tenure', 'property_type']])
                X = self.scaler.transform(X)

                K_neighbours = neigh.kneighbors(X)

                sample_size = 11

                recommendations_idx = K_neighbours[1].tolist()[0]

                recommended_df = self.all_X.iloc[recommendations_idx]

                recommended_df_transformed = np.take(self.X_to_compare, recommendations_idx, axis=0)
                avg_pairwise_distance = compute_average_distance(recommended_df_transformed)

                original_price = original_X["price"]
                upper_bound_price = 1.2 * original_price
                lower_bound_price = 0.8 * original_price
                original_area = original_X["size_sqft"]
                upper_bound_area = 1.2 * original_area
                lower_bound_area = 0.8 * original_area

                count_price_in_range = 0
                count_area_in_range = 0

                for i in range(1, len(recommended_df)):
                    if (recommended_df.iloc[i]["price"] <= upper_bound_price) and (
                            recommended_df.iloc[i]["price"] >= lower_bound_price):
                        count_price_in_range += 1
                    if (recommended_df.iloc[i]["size_sqft"] <= upper_bound_area) and (
                            recommended_df.iloc[i]["size_sqft"] >= lower_bound_area):
                        count_area_in_range += 1
                pct_price_in_range = count_price_in_range / sample_size * 100
                pct_area_in_range = count_area_in_range / sample_size * 100

                score = pct_price_in_range + pct_area_in_range

                for feature in ["property_type", "tenure", "num_beds", "num_baths", "planning_area", "subzone"]:
                    original = original_X[feature]
                    pct_correct = (len(recommended_df[recommended_df[feature] == original])-1) / sample_size * 100
                    score += pct_correct

            # current listing is a DBScan cluster
            else:
                # get index of listings in same cluster
                idx_same_cluster = np.where(dbscan.labels_ == target_cluster)[0]
                idx_same_cluster = idx_same_cluster[idx_same_cluster != i]  # remove target index from 10 recommendations in cluster
                
                # some clusters have less than 10 elements, take min of cluster size and 10
                sample_size = min(len(idx_same_cluster), 10)
                
                # select random sample_size number of listing from cluster
                recommendations_idx = np.random.choice(idx_same_cluster, sample_size, replace=False)
                
                # get recommendations df
                recommended_df = self.all_X.iloc[recommendations_idx]

                # recommended_df in numpy array form for distance calculation
                recommended_df_transformed = np.take(self.X_to_compare, np.append(recommendations_idx, [i]), axis=0)
                avg_pairwise_distance = compute_average_distance(recommended_df_transformed)

                original_price = original_X["price"]
                upper_bound_price = 1.2*original_price
                lower_bound_price  = 0.8*original_price
                original_area = original_X["size_sqft"]
                upper_bound_area = 1.2*original_area
                lower_bound_area  = 0.8*original_area

                count_price_in_range = 0
                count_area_in_range = 0

                for i in range(len(recommended_df)):
                    if (recommended_df.iloc[i]["price"] <= upper_bound_price) and (recommended_df.iloc[i]["price"] >= lower_bound_price):
                        count_price_in_range += 1
                    if (recommended_df.iloc[i]["size_sqft"] <= upper_bound_area) and (recommended_df.iloc[i]["size_sqft"] >= lower_bound_area):
                        count_area_in_range += 1
                pct_price_in_range = count_price_in_range / sample_size * 100
                pct_area_in_range = count_area_in_range / sample_size * 100

                score = pct_price_in_range + pct_area_in_range

                for feature in ["property_type", "tenure", "num_beds", "num_baths", "planning_area", "subzone"]:
                    original = original_X[feature]
                    pct_correct = len(recommended_df[recommended_df[feature]==original])/ sample_size *100
                    score += pct_correct

            total_scores += score/8 # average alignment score for each 8 property attributes
            total_distance += avg_pairwise_distance

        return total_scores/num_records, total_distance/num_records

        
    def transform(self, idx, k=10, verbose=False):

        """
        Get top k recommendations for row input.

        """

        original_X = self.all_X.iloc[idx]

        dbscan = DBSCAN(eps=0.3, min_samples=k+1)

        dbscan.fit(self.X_to_compare)

        target_cluster = dbscan.labels_[idx]

        if verbose:
            print("dbscan.labels_: ", dbscan.labels_)
            print("target_cluster: ", target_cluster)

            (unique, counts) = np.unique(dbscan.labels_, return_counts=True)
            frequencies = np.asarray((unique, counts)).T

            print("number of elements in each cluster: ")
            print(frequencies)

        if target_cluster==-1:
            if verbose:
                print("Item detected as outlier by DBScan. Performing KNN.")

            # fit KNN model
            neigh = NearestNeighbors(n_neighbors=k+1)
            neigh.fit(self.X_to_compare)

            # transform input row to numerical columns
            X = pd.DataFrame(
                data=[original_X.values],
                columns=self.all_columns
            )

            X = X[self.feature_columns]

            X[['tenure', 'property_type']] = self.enc.transform(X[['tenure', 'property_type']])
            X = self.scaler.transform(X)

            # find K nearest neighbours (k+1 because KNN includes current listing)
            K_neighbours = neigh.kneighbors(X)
            recommendations_idx = K_neighbours[1].tolist()[0]
            recommended_df = self.all_X.iloc[recommendations_idx]
            recommended_df_out = recommended_df[self.show_columns]

        else:
            idx_same_cluster = np.where(dbscan.labels_ == target_cluster)[0]
            idx_same_cluster = idx_same_cluster[idx_same_cluster != idx]  # remove target index from 10 recommendations in cluster

            sample_size = min(len(idx_same_cluster), k)

            recommendations_idx = np.random.choice(idx_same_cluster, sample_size, replace=False)

            recommended_df = self.all_X.iloc[recommendations_idx]
            recommened_df_out = recommended_df[self.show_columns]

        if verbose:
            print("========================== Alignment ========================== ")
            original_price = original_X["price"]
            upper_bound_price = 1.2*original_price
            lower_bound_price  = 0.8*original_price
            original_area = original_X["size_sqft"]
            upper_bound_area = 1.2*original_area
            lower_bound_area  = 0.8*original_area

            count_price_in_range = 0
            count_area_in_range = 0

            for i in range(len(recommended_df)):
                if (recommended_df.iloc[i]["price"] <= upper_bound_price) and (recommended_df.iloc[i]["price"] >= lower_bound_price):
                    count_price_in_range += 1
                if (recommended_df.iloc[i]["size_sqft"] <= upper_bound_area) and (recommended_df.iloc[i]["size_sqft"] >= lower_bound_area):
                    count_area_in_range += 1
            pct_price_in_range = count_price_in_range / sample_size * 100
            pct_area_in_range = count_area_in_range / sample_size * 100
            print(f"Price (within +-20%): {pct_price_in_range} %")
            print(f"Area Sqrt (within +-20%): {pct_area_in_range}%")

            original_planning_area = original_X["planning_area"]
            pct_correct_planning_area = len(recommended_df[recommended_df["planning_area"] == original_planning_area]) / sample_size * 100
            print(f"planning_area (same): {pct_correct_planning_area}%")

            original_subzone= original_X["subzone"]
            pct_correct_subzone = len(recommended_df[recommended_df["subzone"] == original_subzone]) / sample_size * 100
            print(f"subzone (same): {pct_correct_subzone}%")

            stats_features = [x for x in self.feature_columns if x not in ['size_sqft', 'price', 'subzone_area_size',
                                                          'subzone_population', 'planning_area_area_size',
                                                          'planning_area_population', 'planning_area_population_density',
                                                          'subzone_population_density']]

            for feature in stats_features:
                original = original_X[feature]
                pct_correct = len(recommended_df[recommended_df[feature]==original])/ sample_size *100
                print(f"{feature} (same): {pct_correct}%")


        return recommened_df_out