import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

class PreprocessorData:

    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = self.load_data()

    # Load data from csv file
    def load_data(self):
        return pd.read_csv(self.file_path)

    def clean_data(self) :
        # Check how many rows of each attribute are NaN
        #print ((df.isnull().sum()).sort_values(ascending=False)) 
        df = self.dataset
        if (df.isnull().sum().sum()) != 0:

            #Replace empty strings if any.
            df.replace('', np.nan, inplace=True) 

            #Make NaN as the placeholder for every null value representation
            df.fillna(value=np.nan, inplace=True)

            #Drop columns with low price corr  and delete rows with high MISSING values
            df = df.drop(columns=['locality','epc','region','latitude','longitude',
                                'cadastral_income','subproperty_type','fl_open_fire',
                                'construction_year','fl_double_glazing','id', 
                                'fl_floodzone', 'equipped_kitchen','fl_furnished','fl_garden','fl_terrace','fl_swimming_pool'])
            cleand_df = df[~df['heating_type'].isin(['MISSING'])]
            cleand_df = cleand_df[~cleand_df['state_building'].isin(['MISSING'])]
            cleand_df = cleand_df[~cleand_df['primary_energy_consumption_sqm'].isna()]

        
            print(cleand_df.shape[0])
            print(cleand_df.shape[1])
            self.dataset = cleand_df
            return self.dataset 
  
    def preprocessing_data(self, cleand_df):
        self.preprocessing_numeric_data()
        self.preprocessing_Categorical_data(cleand_df)
        
      

    #SimpleImputer 
    def preprocessing_numeric_data(self):
        cleand_df = self.dataset
        numeric_df = cleand_df.select_dtypes(include='number')

        # Logical zero for Apartment not mean
        numeric_df.fillna({'surface_land_sqm': 0}, inplace=True)

        # Impute missing values in each column separately
        imputer = SimpleImputer(strategy='mean')
        numeric_df = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)

        # Check how many rows of each attribute are NaN
        #print ((numeric_df.isnull().sum()).sort_values(ascending=False))
        self.dataset = numeric_df
       
    
    #OneHotEncoder for property_type
    #OrdinalEncoder for state_building
    #Explicitly encode heating type
    def preprocessing_Categorical_data(self,cleand_df):
        numeric_df = self.dataset
        #OneHotEncoder for property_type
        property_type = cleand_df[["property_type"]]
        enc = OneHotEncoder(drop='first', sparse_output=False).set_output(transform="pandas")
        property_type_encoded = enc.fit_transform(property_type)

        # Make sure indexes match before joining to avoid NaN when joining
        property_type_encoded.index = numeric_df.index 
        numeric_df = numeric_df.join(property_type_encoded)
        numeric_df.rename(columns = {'property_type_HOUSE':'property_type'}, inplace = True)
        #print(f'Check df after enc property_type {numeric_df.isnull().sum()}')


        province = cleand_df[["province"]]
        enc = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
        province_encoded = enc.fit_transform(province)

        # Make sure indexes match before joining to avoid NaN when joining
        province_encoded.index = numeric_df.index 
        numeric_df = numeric_df.join(province_encoded)

        #OrdinalEncoder for state_building
        building_state = cleand_df[['state_building']]
        building_state_hierarchy = [
        'TO_RESTORE',
        'TO_BE_DONE_UP',
        'TO_RENOVATE',
        'JUST_RENOVATED',
        'GOOD',
        'AS_NEW'
        ]

        encoder = OrdinalEncoder(categories=[building_state_hierarchy])
        building_state_encoded= encoder.fit_transform(building_state)

        building_state_encoded_df = pd.DataFrame(
            building_state_encoded, columns=['state_building'], index=numeric_df.index
        )

        numeric_df = numeric_df.join(building_state_encoded_df)
        #print(f'Check df after enc state_building {numeric_df.isnull().sum()}')

        energy_order = {
        'CARBON': 0,
        'WOOD': 1,
        'PELLET': 2,
        'FUELOIL': 3,
        'GAS': 4,
        'ELECTRIC': 5,
        'SOLAR': 6,
        }

        heat_type_encoded = cleand_df['heating_type']
        heat_type_encoded= cleand_df['heating_type'].map(energy_order)
        heat_type_encoded.index = numeric_df.index 
        numeric_df = numeric_df.join(heat_type_encoded)
        #print(f'Check df after enc heating_type {numeric_df.isnull().sum()}')

        self.dataset = numeric_df

    # Clean and Preproccess data
    def preprocess_data(self):
       
        cleand_df = self.clean_data()
        self.preprocessing_data(cleand_df) 
        return self.dataset       
        