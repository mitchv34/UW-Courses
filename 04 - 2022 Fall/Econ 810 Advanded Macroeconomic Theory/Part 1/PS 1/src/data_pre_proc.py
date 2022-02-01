import pandas as pd
import numpy as np

def read_data(path):
    '''
    Reads in the data from the given path or url.
    '''
    data = pd.read_stata(path)
    return data

def filter_data(data, start_year, end_year, variables ):
    '''
    Filters the data for the given start and end year.
    Inputs:
        data: pandas dataframe
        start_year: num
        end_year: num
        variables: list or dict

    Returns:
        df: pandas dataframe
    '''
    # First we drop individuals from the SEO oversample
    df = data[data.x11104LL == 'Main Sample    11']
    # Then we filter the dataframe by by and years
    df = df[(df.year >= start_year) & (df.year <= end_year)]
    
    if isinstance(variables, list):
        # If the variables are a list, we return the dataframe with the variables
        df = df[variables]
    elif isinstance(variables, dict):
        # If the variables are a dict, we return the dataframe with the variables and rename the columns
        # Create a new dataframe with the data we want
        df = df[variables.keys()]
        # Rename the columns
        df.rename(columns=variables, inplace=True)
    return df 

def filter_by_earnigs(data, earnings_variable, earnings_cuttof, number_of_years):
    '''
    Filters the data for the given earnings_variable and earnings_cuttof and number_of_years.
    Inputs:
        data: pandas dataframe
        earnings_variable: str
        earnings_cuttof: num
        number_of_years: num
    Outputs:
        df: pandas dataframe
    '''

    # Next we create a summary for any individual how many years has earned more than the cutoff
    summary = data.groupby("id").agg({earnings_variable : lambda column : column[column >= earnings_cuttof].count()})
    # The we select the individuals who have earned more than the cutoff in `n_years` number of years
    id_s = summary[summary[earnings_variable] >= number_of_years].index
    # Set Personal id as the index
    df = data.set_index("id")
    # Then we filter the dataframe by the summary
    df = df.loc[id_s]
    # Finally we reset the index and return the dataframe
    # df.reset_index(inplace=True)
    return df
