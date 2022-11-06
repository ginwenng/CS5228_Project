from category_encoders.ordinal import OrdinalEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors

from scipy.spatial.distance import pdist
import numpy as np

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

class KNNRecommender:
    """
    Use KNN algorithm to find the top k similar listing to the input listing. 

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
        
        self.X_to_compare = X # transformed X
        
        return self

    def get_overall_alignment_score(self, full_df):
        
        """
        Get top 10 recommendations for each row in input dataframe and calculate alignment score and
        average distance among target + recommendation points.

        """

        num_records = len(full_df)
        total_scores = 0
        total_distance = 0

        # Fit nearestNeighbours (11 including target row)
        neigh = NearestNeighbors(n_neighbors=11)
        neigh.fit(self.X_to_compare)

        for i in range(num_records):
            score=0
            original_X = full_df.iloc[i]

            # tranform row
            X = pd.DataFrame(
                data=[original_X.values],
                columns=self.all_columns
            )
            X = X[self.feature_columns]
            X[['tenure', 'property_type']] = self.enc.transform(X[['tenure', 'property_type']])
            X = self.scaler.transform(X)

            # get 11 nearest neighbours for row
            K_neighbours = neigh.kneighbors(X)
            recommendations_idx = K_neighbours[1].tolist()[0]
            recommended_df = self.all_X.iloc[recommendations_idx]

            # recommended_df in numpy array form for distance calculation
            recommended_df_transformed = np.take(self.X_to_compare, recommendations_idx, axis=0)
            avg_pairwise_distance = compute_average_distance(recommended_df_transformed)

            # calculate alignment score for recommendations
            original_price = original_X["price"]
            upper_bound_price = 1.2*original_price
            lower_bound_price  = 0.8*original_price
            original_area = original_X["size_sqft"]
            upper_bound_area = 1.2*original_area
            lower_bound_area  = 0.8*original_area

            count_price_in_range = 0
            count_area_in_range = 0
            for i in range(1, len(recommended_df)):
                if (recommended_df.iloc[i]["price"] <= upper_bound_price) and (recommended_df.iloc[i]["price"] >= lower_bound_price):
                    count_price_in_range += 1
                if (recommended_df.iloc[i]["size_sqft"] <= upper_bound_area) and (recommended_df.iloc[i]["size_sqft"] >= lower_bound_area):
                    count_area_in_range += 1
            pct_price_in_range = count_price_in_range / 10 * 100
            pct_area_in_range = count_area_in_range / 10 * 100

            score = pct_price_in_range + pct_area_in_range

            for feature in ["property_type", "tenure", "num_beds", "num_baths", "planning_area", "subzone"]:
                original = original_X[feature]
                pct_correct = (len(recommended_df[recommended_df[feature]==original])-1)/ 10 *100
                score += pct_correct

            total_scores += score/8 # average alignment score for each 8 property attributes
            total_distance += avg_pairwise_distance


        return total_scores/num_records, total_distance/num_records

        
    def transform(self, X, k=10, verbose=False):

        """
        Get top k recommendations for row input.

        """

        original_X = X

        # transform row into numerical attributes
        X = pd.DataFrame(
            data=[X.values],
            columns=self.all_columns
        )
        X = X[self.feature_columns]
        X[['tenure', 'property_type']] = self.enc.transform(X[['tenure', 'property_type']])
        X = self.scaler.transform(X)

        # fit KNN model
        neigh = NearestNeighbors(n_neighbors=k+1)
        neigh.fit(self.X_to_compare)

        # get neighbours of row
        K_neighbours = neigh.kneighbors(X)
        recommendations_idx = K_neighbours[1].tolist()[0]
        recommended_df = self.all_X.iloc[recommendations_idx]
        recommended_df_out = recommended_df[self.show_columns]

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

            for i in range(1, len(recommended_df)):
                if (recommended_df.iloc[i]["price"] <= upper_bound_price) and (recommended_df.iloc[i]["price"] >= lower_bound_price):
                    count_price_in_range += 1
                if (recommended_df.iloc[i]["size_sqft"] <= upper_bound_area) and (recommended_df.iloc[i]["size_sqft"] >= lower_bound_area):
                    count_area_in_range += 1
            pct_price_in_range = count_price_in_range / k * 100
            pct_area_in_range = count_area_in_range / k * 100
            print(f"Price (within +-20%): {pct_price_in_range} %")
            print(f"Area Sqrt (within +-20%): {pct_area_in_range}%")

            original_planning_area = original_X["planning_area"]
            pct_correct_planning_area = (len(recommended_df[recommended_df["planning_area"] == original_planning_area])-1) / k * 100
            print(f"planning_area (same): {pct_correct_planning_area}%")

            original_subzone= original_X["subzone"]
            pct_correct_subzone = (len(recommended_df[recommended_df["subzone"] == original_subzone])-1) / k * 100
            print(f"subzone (same): {pct_correct_subzone}%")

            stats_features = [x for x in self.feature_columns if x not in ['size_sqft', 'price', 'subzone_area_size',
                                                          'subzone_population', 'planning_area_area_size',
                                                          'planning_area_population', 'planning_area_population_density',
                                                          'subzone_population_density']]

            for feature in stats_features:
                original = original_X[feature]
                pct_correct = (len(recommended_df[recommended_df[feature]==original])-1)/ k *100
                print(f"{feature} (same): {pct_correct}%")

        return recommended_df_out[1:]