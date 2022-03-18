__author__ = "Gurdeep Singh Bhambra"
__shayari__ = '''
                Kin baaton ko pehle bolun to theek rahega,
                kuch soyi yaadein nahi uthani hai,
                khafa hun ke nahi mujhe yeh baatein nahi batani hai.
              '''

import pandas as pd
import numpy as np
import datetime


class NanAnalyzer:
    '''
        Class: NanAnalyzer
        Description: Used to analyze and collect partially, fully nan columns in a dataframe
    '''
    def __init__(self, df):
        if(type(df) != type(pd.DataFrame())):
            raise TypeError("NanAnalyzer Object expects a pandas dataframe object")
        elif(df.empty):
            raise ValueError("Empty DataFrame given")
        elif(df.shape[0] == 0):
            raise ValueError("Dataframe has a rows")
        
        self._df = df.copy()
        self._only_nan_cols, self._partial_nan_cols, self._no_nan_cols = [], [], []
        self._partial_nan_count_df = pd.DataFrame({})
        self._total_nan_values = 0
        self._total_partial_nan_values=0
        self.checkNaNValues()
    
    def checkNaNValues(self):
        partial_nan_count=0
        partial_col_nan_count_dict = {'Columns':[], 'Partial_NaN_Count':[]}
        for col in self._df.columns:
            if(pd.isnull(self._df.loc[:, col]).all()):
                self._only_nan_cols.append(col)
                self._total_nan_values+=self._df.loc[:, col].shape[0]
            elif(pd.isnull(self._df.loc[:, col]).any()):
                self._partial_nan_cols.append(col)
                partial_nan_count = self._df.loc[:, col].isnull().sum()
                self._total_partial_nan_values+=partial_nan_count
                self._total_nan_values+=partial_nan_count
                partial_col_nan_count_dict['Columns'].append(col)
                partial_col_nan_count_dict['Partial_NaN_Count'].append(partial_nan_count)
            else:
                self._no_nan_cols.append(col)
        self._partial_nan_count_df = pd.DataFrame(partial_col_nan_count_dict)
        self._partial_nan_count_df['NaN_Vals_By_Total_Vals'] = np.round(self._partial_nan_count_df['Partial_NaN_Count']/self._df[self._df.columns[0]].shape[0], 3)
        
    def printNaNInfo(self):
        print("\nNAN INFO IN THE DATASET:")
        print("\nNo of columns with only NaN values:", len(self._only_nan_cols))
        print("Column list with only NaN values:", self._only_nan_cols, sep="\n")
        print("\nNo of columns with partial NaN values:", len(self._partial_nan_cols))
        print("Column list with partial NaN values:", self._partial_nan_cols, sep="\n")
        print("\nNo of columns with no NaN values:", len(self._no_nan_cols))
        print("Column list with no NaN values:", self._no_nan_cols, sep="\n")
        print("\nTotal Values in dataset:", self._df.shape[0]*self._df.shape[1])
        print("\nTotal NaN Values in the dataset:", self._total_nan_values)
        print("\nTotal Partial NaN Values in the dataset:", self._total_partial_nan_values)
        print("\nPartial NaN Values Info:", self._partial_nan_count_df, sep="\n")
        print("\nTotal NaN Values/Total Datset Values:", round(np.asarray(self._total_nan_values)/(self._df.shape[0]*self._df.shape[1])*100, 2), "%")
    
    def getDict(self):
        ret_dict = {"only_nan_cols":self._only_nan_cols,
                    "partial_nan_cols":self._partial_nan_cols,
                    "no_nan_cols":self._no_nan_cols,
                    "total_nan_values":self._total_nan_values,
                    "total_partial_nan_values":self._total_partial_nan_values,
                    "partial_nan_count_df":self._partial_nan_count_df}
        return ret_dict
    
class DuplicateAnalyzer:
    '''
        Class: DuplicateAnalyzer
        Description: Finds the duplicate columns and rows
    '''
    def __init__(self, df, partial_nan_cols):
        if(type(df) != type(pd.DataFrame())):
            raise TypeError("NanAnalyzer Object expects a pandas dataframe object")
        elif(df.empty):
            raise ValueError("Empty DataFrame given")
        elif(df.shape[0] == 0):
            raise ValueError("Dataframe has a rows")
        if(type(partial_nan_cols) != type(list())):
            raise TypeError("NanAnalyzer Object expects a python list object")
        elif(int(len(partial_nan_cols)) == 0):
            raise ValueError("Empty list of partial_nan_cols given")
        elif(np.asarray(partial_nan_cols).dtype.kind not in ['U']):
            raise ValueError("partial_nan_cols should only contain strings")

        self._df = df.copy()
        self._cols = partial_nan_cols
        self._similar_partial_nan_cols = {col_name:[] for col_name in partial_nan_cols}
        self._duplicate_rows = list()
        self.checkDuplicateCols()
        self.checkDuplicateRows()
        
    def checkDuplicateCols(self):
        for i in range(int(len(self._cols))):
            curr_col_to_compare_with_others = self._cols[i]
            for j in range(int(len(self._cols))):
                if(i==j):
                    continue
                curr_col_to_be_compared = self._cols[j]
                if(self._df[curr_col_to_compare_with_others].equals(self._df[curr_col_to_be_compared])):
                    self._similar_partial_nan_cols[curr_col_to_compare_with_others].append(curr_col_to_be_compared)
        
        self._similar_partial_nan_cols = {key:val for key, val in self._similar_partial_nan_cols.items() if(int(len(val))!=0)}
        #self._similar_partial_nan_cols = {key:val for key, val in self._similar_partial_nan_cols.items() if("norm" not in key)}
    
    def checkDuplicateRows(self):
        self._duplicate_rows = self._df[self._cols].duplicated()
    
    def printDuplicateInfo(self):
        print("\nMatching columns count:", len(self._similar_partial_nan_cols))
        print("\nMatching columns dict:", self._similar_partial_nan_cols, sep="\n")
        print("\nDuplicate Row count:", self._duplicate_rows.sum()) 
        print("\nDUplicate Rows:")
        return self._df[self._duplicate_rows]
    
    def getDict(self):
        ret_dict = {'duplicate_col_list':self._similar_partial_nan_cols,
                    'duplicate_row_list':self._duplicate_rows}
        return ret_dict

class DataTypeAnalyzer:
    '''
        Class: DataTypeAnalyzer
        Desc: Analyzes columns of a Dataframe for all types of suitable DataTypes and converts it.
    '''
    SUPPORTED_DTYPES = ['int64', 'float64', 'datetime', 'object']
    SUPPORTED_DTYPES_FUNCTIONS = {'int64':int, 'float64':float, 'datetime':pd.to_datetime, 'str':str}
    
    @staticmethod
    def __checkDataframe(df):
        if(type(df) != type(pd.DataFrame())):
            raise TypeError("DataTypeAnalyzer Object expects a pandas dataframe object")
        elif(df.empty):
            raise ValueError("Empty DataFrame given")
        elif(df.shape[0] == 0):
            raise ValueError("Dataframe has a rows")
        return df.copy()
    
    @staticmethod
    def __isDateCol(_df, col):
        try:
            pd.to_datetime(_df[col])
            return True
        except Exception:
            pass
        return False
    
    @staticmethod
    def __isInt64Col(_df, col):
        try:
            np.int64(_df[col])
            return True
        except Exception:
            pass
        return False
    
    @staticmethod
    def __isFloat64Col(_df, col):
        try:
            np.float64(_df[col])
            return True
        except Exception:
                pass
        return False

    def dtypeInfo(self, df):
        df = self.__checkDataframe(df)
        info_dict = {key:list() for key in df.columns}
        fillna_with = '-'
        for col in info_dict.keys():
            if(self.__isInt64Col(df, col)):
                info_dict[col].append('int64')
            else:
                info_dict[col].append(fillna_with)
            if(self.__isFloat64Col(df, col)):
                info_dict[col].append('float64')
            else:
                info_dict[col].append(fillna_with)
            if(self.__isDateCol(df, col)):
                info_dict[col].append('datetime')
            else:
                info_dict[col].append(fillna_with)
            info_dict[col].append('object')
        return pd.DataFrame(info_dict)
    
    def colMisMatchDtypeInfo(self, col_series, expected_dtype):
        if(not isinstance(col_series, pd.Series)):
            raise TypeError('Pandas Series Expected')
        if(not isinstance(expected_dtype, str)):
            raise TypeError('String Expected for expected_dtype argument')
        if(expected_dtype not in list(DataTypeAnalyzer.SUPPORTED_DTYPES_FUNCTIONS.keys())):
            raise ValueError('Available Dtypes:', list(DataTypeAnalyzer.SUPPORTED_DTYPES_FUNCTIONS.keys()))
        mismatch_values = []
        for val in col_series.unique():
            try:
                DataTypeAnalyzer.SUPPORTED_DTYPES_FUNCTIONS[expected_dtype](val)
            except Exception:
                mismatch_values.append(val)
        return mismatch_values
           
    def findUnExpectedDtypeValues(self, df, expected_dtype_dict):
        df = self.__checkDataframe(df)
        if(not (isinstance(expected_dtype_dict, dict))):
            raise TypeError("Expected a python dict or None")
        if(int(len(set(expected_dtype_dict.values()) - set(list(DataTypeAnalyzer.SUPPORTED_DTYPES_FUNCTIONS.keys())))) != 0):
            raise ValueError("Dictionary must have supported dtypes\nSupported dtypes:", DataTypeAnalyzer.SUPPORTED_DTYPES)
        
        res_dict = {'Columns':list(), 'ExpectedDtype':list(), 'MismatchedValues':list()}
        
        for col, exp_dtype in expected_dtype_dict.items():
            res_dict['Columns'].append(col)
            res_dict['ExpectedDtype'].append(exp_dtype)
            res_dict['MismatchedValues'].append(self.colMisMatchDtypeInfo(df[col], exp_dtype))
        
        return pd.DataFrame(res_dict)
        
    def convertDtypes(self, df, dtype_dict = None, dayfirst=False):
        df = self.__checkDataframe(df)
        if(not (isinstance(dtype_dict, type(None)) or (isinstance(dtype_dict, dict)))):
            raise TypeError("Expected a python dict or None")
        if(int(len(set(dtype_dict.keys()) - set(df.columns.values.tolist()))) != 0):
            raise IndexError("Dictionary must specify dtype for all the columns")
        if(int(len(set(dtype_dict.values()) - set(DataTypeAnalyzer.SUPPORTED_DTYPES))) != 0):
            raise ValueError("Dictionary must have supported dtypes\nSupported dtypes:", DataTypeAnalyzer.SUPPORTED_DTYPES)
        for col in df.columns:
            curr_col_dtype = dtype_dict[col]
            if(curr_col_dtype == 'datetime'):
                df[col] = pd.to_datetime(df[col], dayfirst=dayfirst)
            elif(curr_col_dtype == 'object'):
                df[col] = df[col].map(lambda x: str(x))
            else:
                df[col] = df[col].astype(curr_col_dtype)
        return df

class DateTimeAnalyzer:
    '''
        Class: DateTimeAnalyzer
        Desc: Converts any column of object dtype to datetime if its of the supported format.
    '''
    def __init__(self, df):
        if(type(df) != type(pd.DataFrame())):
            raise TypeError("NanAnalyzer Object expects a pandas dataframe object")
        elif(df.empty):
            raise ValueError("Empty DataFrame given")
        elif(df.select_dtypes(['object']).columns.shape[0] == 0):
            raise TypeError("Dataframe has no object dtype columns")
        self._df = df.copy()
        self._non_date_cols = list()
        self._date_cols = list()
        self._ret_dict = dict()
        self.checkDateColumns()

    def checkDateColumns(self):
        for c in self._df.select_dtypes(['object']).columns:
            try:
                self._df[c] = pd.to_datetime(self._df[c])
                self._date_cols.append(c)
            except Exception:
                self._non_date_cols.append(c)
        self._ret_dict['object_cols'] = self._df.select_dtypes(['object']).columns
        self._ret_dict['non_date_cols'] = self._non_date_cols
        self._ret_dict['date_cols'] = self._date_cols

    def printDateInfo(self):
        print("Total Object dtype columns:", len(self._ret_dict['object_cols']))
        print("Total Date Columns:", len(self._ret_dict['date_cols']))
        print("Date columns list:", self._ret_dict['date_cols'])
        print("Total Non Date Columns:", len(self._ret_dict['non_date_cols']))

    def convertedDataframe(self):
        return self._df

    def retDict(self):
        return self._ret_dict
                    
class Utils:
    '''
        Class: Utils
        Description: Utils class having some functions wrapped
    '''
    @staticmethod
    def outliers(df, iqr_multiplier=1.5):
        '''
            Desc: Function to check for outliers in each column of the dataset.
                  Outliers are checked using box-whisker

            Args: df,
                  iqr_multiplier=1.5
        '''
        stats = df.describe()
        outlier_index_dict = dict()
        for col in stats.columns:
            iqr = stats[col].loc['75%'] - stats[col].loc['25%']
            top_whisker = stats[col].loc['75%'] + iqr_multiplier * iqr
            bottom_whisker = stats[col].loc['25%'] - iqr_multiplier * iqr
            #print(top_whisker, bottom_whisker)
            indexes = df[~((bottom_whisker <= df[col]) & (df[col] <= top_whisker))].index
            if(not indexes.empty):
                outlier_index_dict[col] = indexes
        return outlier_index_dict

    
class FileParserTools:
    '''
        Class: FileParserTools
        Desc: Reads a delimited file and provides info on it.
    '''
    def __init__(self, filename, delimiter=',', is_byte_file=False, encoding='utf-8'):
        self._filename = filename
        self._delimiter = delimiter
        self._encoding = encoding
        self._filetype = 'r'
        if(is_byte_file):
            self._filetype = 'rb'
        self._col_line_count = None
        self._major_col_count = None

    def rowShiftInfo(self):
        with open(self._filename, self._filetype, encoding=self._encoding) as f:
            col_line_count = dict()
            for line in f.readlines():
                curr_col_count = int(len(line.split(self._delimiter)))
                if(curr_col_count not in col_line_count):
                    col_line_count[curr_col_count] = 1
                else:
                    col_line_count[curr_col_count] = col_line_count[curr_col_count] + 1
            self._col_line_count = col_line_count
            self._major_col_count = max(self._col_line_count.items(), key=lambda item: item[1])[0]
            return pd.DataFrame(col_line_count.items(), columns=['No of Columns', 'No of Rows'])

    def printOddColumnCountRow(self, skip_col_count=[]):
        data = []
        with open(self._filename, self._filetype, encoding=self._encoding) as f:
            for line in f.readlines():
                if(int(len(line.split(self._delimiter))) not in skip_col_count):
                    data.append(line.split(self._delimiter))
        return data

    def extractColumn(self, col_index):
        col_data = []
        with open(self._filename, self._filetype, encoding=self._encoding) as f:
            for line in f.readlines():
                col_data.append(line.split(self._delimiter)[col_index])
        return col_data

    def getColumns(self, col_at_row_index=0):
        with open(self._filename, self._filetype, encoding=self._encoding) as f:
            for row_index, line in enumerate(f.readlines()):
                if(row_index == col_at_row_index):
                    cols = line.split(self._delimiter)
                    return list(zip(list(range(int(len(cols)))), cols))

    def extractRows(self, col_count, col_names_at_row_index=0):
        data = []
        col_names = []
        with open(self._filename, self._filetype, encoding=self._encoding) as f:
            for row_no, line in enumerate(f.readlines()):
                line_data = line.split(self._delimiter)
                if(row_no == col_names_at_row_index):
                    col_names = line_data
                elif(int(len(line_data)) == col_count):
                    data.append(line_data)
        return pd.DataFrame(data, columns=col_names)