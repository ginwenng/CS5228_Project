import pandas as pd
import numpy as np
import re

class DataProcessor:
    
    def __init__(self):
        
        self.current_year = 2022
        self.hdb_tenure_dict = dict({'hdb': '99-year leasehold'}) # fill na for hdb to "99-year leasehold"
        
        self.columns_to_drop = ["elevation", "floor_level", "furnishing", "available_unit_types", "total_num_units"]
        
        self.pairs = [
            ('built_year', 'property_name'),
            ('built_year', 'address_'),
            ('tenure', 'address_'),
            ('subzone', 'address_'),
            ('tenure', 'subzone'),
            ('built_year', 'subzone'),
            ('num_beds', 'name_size'),
            ('num_beds', 'type_size'),
            ('num_baths', 'name_size'),
            ('num_baths', 'type_size'),
            ('num_baths', 'num_beds'),
        ]
        self.mapper_dicts = {}
        
    
    def fit(self, X, y=None):
        
        X = self.drop_redundant_columns(X=X, columns_to_drop=self.columns_to_drop)
        
        X = X[X["price"]!=0] # drop row with zero values
        X = X[X["price"]<1000000000] # drop row with price > $1 billion (not possible for HDB flats)
        
        X = self.clean_property_type(X=X)
        X = self.clean_tenure(X=X)
        X = self.clean_hdb_address(X=X)
        
        X = X[X["size_sqft"]!=0] # drop row with zero values
        X = X[X["size_sqft"]<100000] # drop row with size sqft > 100000
        X = self.clean_size(X=X)
        
        X, intermediate_column_names = self.get_useful_intermediate_columns(X)
        
        for pair in self.pairs:
            feature, learn_from = pair
            mapper_dict = self.get_feature_to_learn_mapping(X=X, feature=feature, learn_from=learn_from)
            if feature =='num_beds' and learn_from == 'type_size':
                mapper_dict.update({
                    'condo_300.0': 1.0,
                    'apartment_300.0': 1.0,
                    'apartment_3800.0': 4.0
                })
            X = self.get_missing_to_treat_with_mapper(X=X, feature=feature, learn_from=learn_from, mapper_dict=mapper_dict)
            self.mapper_dicts.update({pair: mapper_dict})
            
        self.max_built_year = X['built_year'].value_counts().index.sort_values(ascending=False)[0]
        
        X = self.get_built_year_features(X=X)
        
        X = self.drop_redundant_columns(X=X, columns_to_drop=intermediate_column_names)
        X = self.drop_redundant_columns(X=X, columns_to_drop=['planning_area', 'built_year'])
        
        return X
    
    def transform(self, X, y=None):
        
        X = self.drop_redundant_columns(X=X, columns_to_drop=self.columns_to_drop)
        
        X = self.clean_property_type(X=X)
        X = self.clean_tenure(X=X)
        X = self.clean_hdb_address(X=X)
        
        X, intermediate_column_names = self.get_useful_intermediate_columns(X)
        
        for pair in self.pairs:
            feature, learn_from = pair
            mapper_dict = self.mapper_dicts[pair]
            new_mapper_dict = self.get_feature_to_learn_mapping(X=X, feature=feature, learn_from=learn_from)
            new_mapper_dict = self.update_mapping(old_mapping_dict=mapper_dict, new_mapping_dict=new_mapper_dict)
            X = self.get_missing_to_treat_with_mapper(X=X, feature=feature, learn_from=learn_from, mapper_dict=new_mapper_dict)
        
        X = self.get_built_year_features(X=X)
        
        X = self.drop_redundant_columns(X=X, columns_to_drop=intermediate_column_names)
        X = self.drop_redundant_columns(X=X, columns_to_drop=['planning_area', 'built_year'])
        
        return X
        
    def get_useful_intermediate_columns(self, X):
        X['address_'] = X['address'].replace(regex=r'([^a-zA-Z]+?)', value='')
        X['name_size'] = X['property_name'] + '_' + X['size_sqft'].astype(str)
        X['type_size'] = X['property_type'] + '_' + X['size_sqft'].round(-2).astype(str)
        
        intermediate_column_names = ['address_', 'name_size', 'type_size']
        
        return X, intermediate_column_names
    
    def get_feature_to_learn_mapping(self, X, feature, learn_from):
        """
        First, identify the indexes of the records that are missing in the `feature` column.
        Next, for these indexes, identify the values in the `learn_from` column and take the unique, sorted in descending order by the frequency of appearance.
        Then, for each of the value in the `learn_from` column, filter by this value, and get the most frequent `feature` column value.
        Update the mapping dictionary with this value.
        Args:
            X (pd.DataFrame)
            feature (str): The name of the feature to impute.
            learn_from (str): The name of the feature to learn from.
        Returns:
            mapper_dict (dict): Keys represents a value from the `learn_from` column. Values represents the most frequent `feature` column value when the records are filtered using the value from `learn_from` column.
        """
        
        lst = X[X[feature].isna()][learn_from].value_counts().index
        
        mapper_dict = {}

        for itm in lst:
            
            _ = X[X[learn_from]==itm][feature].value_counts().index
            
            if len(_) > 0:
                mapper_dict.update({itm: _[0]})
            
        return mapper_dict
    
    def update_mapping(self, old_mapping_dict, new_mapping_dict):
        for key, value in new_mapping_dict.items():
            if key not in old_mapping_dict.keys():
                old_mapping_dict.update({key: value})
        return old_mapping_dict
    
    def get_missing_to_treat_with_mapper(self, X, feature, learn_from, mapper_dict):
        """
        First, identify the indexes of the records that are missing in the `feature` column.
        Then, using the mapper dictionary to fill the corresponding values that can be found in the mapper dictionary.
        
        Args:
            X (pd.DataFrame)
            feature (str): The name of the feature to impute.
            learn_from (str): The name of the feature to learn from.
            mapper_dict (dict): Keys represents a value from the `learn_from` column. Values represents the most frequent `feature` column value when the records are filtered using the value from `learn_from` column.
        
        Returns:
        X (pd.DataFrame): Dataframe with missing value in column `feature` filled using the mapper_dict.
        
        """
        
        idx = X[X[feature].isna()].index
        X.loc[idx, feature] = X.loc[idx, learn_from].map(mapper_dict)
        
        return X
    
    def drop_redundant_columns(self, X, columns_to_drop):
        X = X.drop(columns=columns_to_drop)
        return X
    
    def clean_property_type(self, X):
        """
        Group all HDBs in property_type and transform to lowercase
        """
        X["property_type"] = X["property_type"].str.lower()
        X["property_type"] = X["property_type"].apply(lambda x: "hdb" if "hdb" in x else x)
        return X
    
    def clean_tenure(self, X):
        """
        Group into "99-year leasehold" and "999-year leasehold" in tenure and fill na
        """
        X["tenure"] = X["tenure"].str.lower()
        X["tenure"] = X["tenure"].apply(lambda x: "999-year leasehold" if x=="freehold" else x) # freehold convert to 999-year leasehold
        X["tenure"] = X["tenure"].apply(
            lambda x: x if pd.isna(x) else (
                "99-year leasehold" if int(str(x).split("-")[0])<200 else "999-year leasehold"
            )
        )
        
        X = self.get_missing_to_treat_with_mapper(X=X, feature='tenure', learn_from='property_type', mapper_dict=self.hdb_tenure_dict)
        
        return X
    
    def clean_hdb_address(self, X):
        """
        Obtain the records that are HDBs.
        Extract the real address of the HDBs from the `title`.
        Then, replace the `address` column with the real address.
        
        Args:
            X (pd.DataFrame): Dataframe prior to replacing the HDB address.
            
        Returns:
            X (pd.DataFrame): Dataframe after replacing the HDB address.
        """
        
        idx = X[X['property_type']=='hdb'].index
        X.loc[idx, 'address'] = X.loc[idx, 'title'].str.split(' in ').str[1].str.split(' ').str[1:].str.join(sep=' ')
        
        return X
        
    
    def clean_size(self, X):

        X["size_sqft"] = X["size_sqft"].apply(
            lambda x: x*10.764 if x<200 else x
        ) # normalise units for outlier values (due to units) from sqm to sqft
        
        return X
    
    def get_built_year_features(self, X):
        
        X['built_year'] = X['built_year'].fillna(self.max_built_year)
        X['age'] = self.current_year - X['built_year']
        X['age'] = X['age'].clip(0)
        X['years_remaining'] = X['tenure'].str.split('-').str[0].astype(int) - X['age']
        
        return X

def clean_data(filepath, train):
    
    df = pd.read_csv(filepath)

    df = df.drop(columns=["elevation"])

    ######### Group all HDBs in property_type and transform to lowercase
    df["property_type"] = df["property_type"].str.lower()
    df["property_type"] = df["property_type"].apply(lambda x: "hdb" if "hdb" in x else x)

    ######### Group into "99-year leasehold" and "999-year leasehold" in tenure and fill na
    df["tenure"] = df["tenure"].str.lower()
    df["tenure"] = df["tenure"].apply(lambda x: "999-year leasehold" if x=="freehold" else x) # freehold convert to 999-year leasehold
    df["tenure"] = df["tenure"].apply(lambda x: x if pd.isna(x) else ("99-year leasehold" if int(str(x).split("-")[0])<200 else "999-year leasehold"))
    missing_tenure = df['tenure'].isna()
    mapping_dict = dict({'hdb': '99-year leasehold'}) # fill na for hdb to "99-year leasehold"
    df.loc[missing_tenure, 'tenure'] = df.loc[missing_tenure, 'property_type'].map(mapping_dict)
    df = df.sort_values(by=["address", "property_type"]) 
    df['tenure'] = df['tenure'].fillna(method='ffill') # fill na by forward fill after sorting by address and property_type


    ######### remove and treat outliers in size_sqft
    if train == True:
        df = df[df["size_sqft"]!=0] # drop row with zero values
        df = df[df["size_sqft"]<100000] # drop row with size sqft > 100000
    df["size_sqft"] = df["size_sqft"].apply(lambda x: x*10.764 if x<200 else x) # normalise units for outlier values (due to units) from sqm to sqft

    ######### forward fill num_beds after sorting by size_sqft
    df = df.sort_values(by="size_sqft")
    df['num_beds'] = df['num_beds'].fillna(method='ffill')

    ######### forward fill num_baths after sorting by size_sqft
    df = df.sort_values(by="size_sqft")
    df['num_baths'] = df['num_baths'].fillna(method='ffill')

    ######### drop built_year, floor_level, furnishing, available_unit_types, total_num_units
    df = df.drop(columns=["built_year", "floor_level", "furnishing", "available_unit_types", "total_num_units"])

    ######### clean lat, lng, subzone, planning_area
    df["address_no_numbers"] = df['address'].str.replace('\d+', '')
    df["address_no_numbers"] = df['address_no_numbers'].str.strip()
    df["address_no_numbers"] = df['address_no_numbers'].str.replace("  ", " ")
    df = df.sort_values(by="address_no_numbers")
    df['subzone'] = df.sort_values(by="address").groupby('address_no_numbers')['subzone'].fillna(method='bfill')
    df['planning_area'] = df.sort_values(by="address").groupby('address_no_numbers')['planning_area'].fillna(method='bfill')
    df = df.drop(columns=["address_no_numbers"])

    ######### clean price
    if train == True:
        df = df[df["price"]!=0] # drop row with zero values
        df = df[df["price"]<1000000000] # drop row with price > $1 billion (not possible for HDB flats)

    df = df.drop(columns=['planning_area'])
    
    return df

def clean_comm_data(df):
    df['type'] = df['type'].replace({'IEPB': 'IEBP'})
    return df