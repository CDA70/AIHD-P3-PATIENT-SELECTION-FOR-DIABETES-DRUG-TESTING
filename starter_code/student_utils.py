import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    print('add column generic_drug_name to the returned dataframe')
    dataframe = pd.merge(df, ndc_df[['Proprietary Name', 'NDC_Code']], left_on = 'ndc_code', right_on = 'NDC_Code')
    dataframe['generic_drug_name'] = dataframe['Proprietary Name']
    dataframe = dataframe.drop(['NDC_Code', 'Proprietary Name'], axis=1)
    return dataframe

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    # initially I used the code "last_encounter_values" in the lesson, but it didn't work. 
    # following the recomendation in the knowledge base pointed me in the right direction (https://knowledge.udacity.com/questions/184962). Also some research in pandas dataframe helped me a lot: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html
    
    first_encounter_df = df.sort_values('encounter_id')
    first_encounter_df = (first_encounter_df.drop_duplicates(subset=['encounter_id'], keep='first').drop_duplicates(subset=['patient_nbr'], keep='first'))
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    df = df.iloc[np.random.permutation(len(df))]
           
    # split train = 60%
    unique_values = df[patient_key].unique()
    train_size = round(len(unique_values) * (1 - 0.4 ))
    print('train size : ', train_size)
    train = df[df[patient_key].isin(unique_values[:train_size])].reset_index(drop=True)
    
    #split validation = 20%
    remaining_values = unique_values[train_size:]
    validation_size = round(len(remaining_values) * 0.5 )
    print('remaining_size : ', validation_size)
    validation = df[df[patient_key].isin(remaining_values[:validation_size])].reset_index(drop=True)
    
    #split test = 20%
    test = df[df[patient_key].isin(remaining_values[validation_size:])].reset_index(drop=True)
   
    print("Total number of unique patients are = ", len(unique_values))
    print("Total number of unique patients in train = ", len(train[patient_key].unique()))
    print("Total number of unique patients in validation = ", len(validation[patient_key].unique()))
    print("Total number of unique patients in test = ", len(test[patient_key].unique()))
    print("Training partition has a shape = ", train.shape) 
    print("Training partition has a shape = ", validation.shape) 
    print("Test partition has a shape = ", test.shape)
    
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        # reference: lesson 3 ex 4: build Embedding Categorical Feature with TF
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file = vocab_file_path, num_oov_buckets=1)
        
        #https://www.tensorflow.org/api_docs/python/tf/feature_column/indicator_column
        tf_categorical_feature_column = tf.feature_column.indicator_column(tf_categorical_feature_column)
        
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    # see lesson 3 and 4 concept: Code for Building Synthetic Dataset
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    
    tf_numeric_feature = tf.feature_column.numeric_column(
        key=col, default_value = default_value, normalizer_fn=normalizer, dtype=tf.float64)
    
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    # see lesson 3 and 4 concept: Building Numerical Feature with TF Feature Column API
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    # Lesson 4: Model evaluation, ... convert_to_binary
    a = df[col]
    a.astype(int)
    return a.apply(lambda x: 1 if x >= 5 else 0 )
    
    
