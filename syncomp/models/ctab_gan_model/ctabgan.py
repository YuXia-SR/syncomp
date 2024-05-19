"""
Generative model training algorithm based on the CTABGANSynthesiser

"""
import pandas as pd
import time
from syncomp.models.ctab_gan_model.pipeline.data_preparation import DataPrep
from syncomp.models.ctab_gan_model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

import warnings

warnings.filterwarnings("ignore")


class CTABGAN():

    def __init__(self,
                 train_df: pd.DataFrame,
                 categorical_columns = [], 
                 log_columns = [],
                 mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                 general_columns = [],
                 non_categorical_columns = [],
                 integer_columns = [],
                 problem_type= {},
                 **synthesizer_config):

        self.__name__ = 'CTABGAN'
              
        self.synthesizer = CTABGANSynthesizer(**synthesizer_config)
        self.train_df = train_df
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
                
    def fit(self):
        
        start_time = time.time()
        self.data_prep = DataPrep(self.train_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.general_columns,self.non_categorical_columns,self.integer_columns,self.problem_type)
        self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], mixed = self.data_prep.column_types["mixed"],
        general = self.data_prep.column_types["general"], non_categorical = self.data_prep.column_types["non_categorical"], type=self.problem_type)
        end_time = time.time()
        print('Finished training in',end_time-start_time," seconds.")


    def generate_samples(self, n_sample):
        
        sample = self.synthesizer.sample(n_sample) 
        sample_df = self.data_prep.inverse_prep(sample)
        
        return sample_df
