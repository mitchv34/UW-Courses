
import pandas as pd
import numpy as np

def read_data(path):
    '''
    Reads in the data from the given path or url.
    '''
    data = pd.read_stata(path)
    return data

def filter_data(data, start_year, end_year, variables, filters = {} ):
    '''
    Filters the data for the given start and end year.
    Inputs:
        data: pandas dataframe
        start_year: num
        end_year: num
        variables: list or dict
        filters: dict {variable: filter} if len(filters) == 1 filter using '==' if len(filters) == 2 filter using '<=' and '>=' 
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
    
    for variable, filter in filters.items():
        # For each variable and filter, we filter the dataframe
        if len(filter) == 1:
            # If the filter is a single number, we filter the dataframe with '=='
            df = df[df[variable] == filter[0]]
        elif len(filter) == 2:
            # If the filter is a list of two numbers, we filter the dataframe with '<=' and '>='
            df = df[(df[variable] >= filter[0]) & (df[variable] <= filter[1])]

    return df 


if __name__ == "__main__":
    data_dict = {
        # Identifiers
        'year' : "year",
        'x11101LL' : 'id',
        'd11101' : 'age',
        # Income variables
        "i11113" : "income"
        }

    # df = read_data('https://www.dropbox.com/s/w65rpy6gj13c02x/pequiv_long.dta?dl=1')
    print("Reading data...")
    df = pd.read_stata('/mnt/c/Users/mitch/Downloads/pequiv_long.dta')
    print("Data read.")
    
    print("Filtering data...")
    df_1 = filter_data(df, 1978, 1997, data_dict, {'age':[18, 71]})

    df_1["cohort"] = df_1.year - df_1.age
    years = np.arange(1978, 1998)
    df_3 = df_1.copy()
    for year in years:
        i_max = df_1[df_1.year == year].income.quantile(.95)
        i_min = df_1[df_1.year == year].income.quantile(.05)
        # print(year, i_max, i_min)
        id_drop = df_1[(df_1.year == year) & ((df_1.income > i_max) | (df_1.income < i_min))]
        # print(year, len(id_drop), len(df_1[(df_1.year == year)]), 100*(len(id_drop)/ len(df_1[(df_1.year == year)])))
        df_3.drop(id_drop.index, inplace=True)

    print("Data filtered.")
    df_3.to_csv("./data/data.csv", index=False)
