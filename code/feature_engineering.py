from data_processor import *
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

def rename_column_names(df):
    df = df.rename(columns={
        'name': 'subzone',
        'population': 'subzone_population',
        'area_size': 'subzone_area_size',
    })
    return df

def generate_area_size_n_population_features(df, subzone_level=True, planning_area_level=True):
    
    if 'subzone' not in df.columns:
        df = rename_column_names(df)
    
    if planning_area_level == True:
        planning_area_level_pop_size = df.groupby('planning_area').sum().reset_index().rename(columns={
            'subzone_area_size': 'planning_area_area_size',
            'subzone_population': 'planning_area_population'
        })
        df = df.merge(planning_area_level_pop_size, on='planning_area', how='left')
        df['planning_area_population_density'] = df['planning_area_population']/df['planning_area_area_size']
    
    if subzone_level == True:
        df['subzone_population_density'] = df['subzone_population']/df['subzone_area_size']
        
    return df

def generate_generic_features(df, feat_df, feature_type, subzone_level=True, planning_area_level=True):
    """
    Compute total number of commercial centres / mrt stations / primary schools / secondary schools / shopping malls in each subzone and planning area.
    
    Args:
        df (pd.DataFrame): Cleaned property-level data (e.g. train.csv)
        feat_df (pd.DataFrame): auxiliary data (e.g. sg-shopping-malls.csv)
        feature_type (str): Name of the parent-level feature to generate
        subzone_level (bool): If subzone_level=True, features at subzone-level will be generated.
        planning_area_level (bool): If planning_area_level=True, features at planning area-level will be generated.
    
    Returns:
        df (pd.DataFrame): Features engineered will be added to the original cleaned dataframe.
    """
    
    if 'subzone' not in df.columns:
        df = rename_column_names(df)
        
    if feature_type == 'num_comm':
        feat_df = clean_comm_data(feat_df)
    
    if subzone_level == True:
        subzone_feat_column = f'subzone_{feature_type}'
        subzone_level_feat_df = feat_df[['subzone', 'name']]\
            .groupby('subzone')\
            .count()\
            .reset_index()\
            .rename(columns={'name': subzone_feat_column})
        df = df.merge(subzone_level_feat_df, on='subzone', how='left')
        df[subzone_feat_column] = df[subzone_feat_column].fillna(0)
    
    if planning_area_level == True:
        planning_area_feat_column = f'planning_area_{feature_type}'
        planning_area_level_feat_df = df[['planning_area', subzone_feat_column]]\
            .groupby('planning_area')\
            .sum()\
            .reset_index()\
            .rename(columns={subzone_feat_column: planning_area_feat_column})
        df = df.merge(planning_area_level_feat_df, on='planning_area', how='left')
        
    return df

def generate_advanced_features(df, feat_df, feature_type, subzone_level=True, planning_area_level=True):
    """
    Compute total number of commercial centres / mrt stations (for each commercial centre type / mrt station line) in each subzone and planning area.
    
    Args:
        df (pd.DataFrame): Cleaned property-level data (e.g. train.csv)
        feat_df (pd.DataFrame): auxiliary data (e.g. sg-commercial-centres.csv)
        feature_type (str): Name of the parent-level feature to generate
        subzone_level (bool): If subzone_level=True, features at subzone-level will be generated.
        planning_area_level (bool): If planning_area_level=True, features at planning area-level will be generated.
    
    Returns:
        df (pd.DataFrame): Features engineered will be added to the original cleaned dataframe.
    """
    
    if 'subzone' not in df.columns:
        df = rename_column_names(df)
    
    df = generate_generic_features(
        df=df, 
        feat_df=feat_df, 
        feature_type=feature_type, 
        subzone_level=subzone_level,
        planning_area_level=planning_area_level
    )
    
    if feature_type == 'num_comm':
        groupby_col = 'type'
        
    elif feature_type == 'num_mrt':
        groupby_col = 'line'
    
    else:
        raise ValueError(f"{feature_type} not supported yet.")
    
    if subzone_level == True:
        subzone_level_feat_df = feat_df.groupby(['subzone', groupby_col])['name']\
            .count()\
            .reset_index()
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(subzone_level_feat_df[[groupby_col]])
        _ = pd.DataFrame(
            data=enc.transform(subzone_level_feat_df[[groupby_col]]).toarray(),
            columns=enc.get_feature_names_out()
        )
        _ = _.mul(subzone_level_feat_df['name'], axis=0)
        subzone_level_feat_df = subzone_level_feat_df.join(_)
        subzone_level_feat_df = subzone_level_feat_df\
            .drop(columns=[groupby_col, 'name'])\
            .groupby('subzone')\
            .sum()\
            .reset_index()
        subzone_level_feat_df.columns = subzone_level_feat_df.columns.str.replace(f'{groupby_col}_', 'subzone_num_')
        columns = subzone_level_feat_df.columns.drop('subzone').tolist()
        df = df.merge(subzone_level_feat_df, on='subzone', how='left')
        df[columns] = df[columns].fillna(0)
    
    if planning_area_level == True:
        columns.append('planning_area')
        planning_area_level_feat_df = df[columns]\
            .groupby('planning_area')\
            .sum()\
            .reset_index()
        planning_area_level_feat_df.columns = planning_area_level_feat_df.columns.str.replace(r'subzone_', 'planning_area_')
        df = df.merge(planning_area_level_feat_df, on='planning_area', how='left')
        
    return df

def generate_all_features(data_directory):
    
    auxiliary_data_path = os.path.join(data_directory, 'auxiliary-data')
    
    subzones = pd.read_csv(os.path.join(auxiliary_data_path, 'sg-subzones.csv'))
    pri_sch = pd.read_csv(os.path.join(auxiliary_data_path, 'sg-primary-schools.csv'))
    sec_sch = pd.read_csv(os.path.join(auxiliary_data_path, 'sg-secondary-schools.csv'))
    malls = pd.read_csv(os.path.join(auxiliary_data_path, 'sg-shopping-malls.csv'))
    comm = pd.read_csv(os.path.join(auxiliary_data_path, 'sg-commerical-centres.csv'))
    mrt = pd.read_csv(os.path.join(auxiliary_data_path, 'sg-mrt-stations.csv'))
    
    subzones = generate_area_size_n_population_features(df=subzones)
    subzones = generate_generic_features(df=subzones, feat_df=pri_sch, feature_type='num_pri_sch')
    subzones = generate_generic_features(df=subzones, feat_df=sec_sch, feature_type='num_sec_sch')
    subzones = generate_generic_features(df=subzones, feat_df=malls, feature_type='num_malls')
    subzones = generate_advanced_features(df=subzones, feat_df=comm, feature_type='num_comm')
    subzones = generate_advanced_features(df=subzones, feat_df=mrt, feature_type='num_mrt')
    
    return subzones
 