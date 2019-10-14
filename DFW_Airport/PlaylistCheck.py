import os
#import time used to track script run time
import pandas as pd
from datetime import datetime

#start = time.time() ### used to track script run time

####################################################################################### XML config section? 
os.chdir(r'S:\mhan\duplicateCheck')

reportDate = datetime.now().strftime("%m-%d-%Y") ## currently set to be current date

#reportTime = '1230'
#reportTime = '1300'
reportTime = '1330'
#reportTime = '1400'
#reportTime = '1500'
#reportTime = '1530'
#reportTime = '1600'

xlsPath = r'.\PFM_{0}_{1}.xlsx'.format(reportDate, reportTime) # dynamically changing xlsPath to pull current date and "current time" from the \\AODBPRDA\PFMHistory
#######################################################################################

# reads the excel file and the playlist sheet only
df = pd.read_excel(xlsPath, 'Playlist Item Load ') 
df.columns = df.iloc[0].str.replace(" ","") # remove spaces from new column headers
df = df[1:].dropna().reset_index(drop = True) # remove first row of column headers and remove NaNs from the dataframe ONLY WORKS IF THE FORMAT OF THE SHEET IS THE SAME
df['DATE'] = df[['Date', 'SchedTime']].apply(lambda x: datetime.combine(*list(x)), axis = 1) # create new date column for unique ID

# creates unique identifier based on hour, minute, location and time of report
xlsEnd = xlsPath.split("_")[2].split('.')[0]
# Unique Identifier format -- playlist initial time:location for playlist:report run time
df['ID'] = (df['DATE'].dt.hour.astype('str').str.pad(width = 2, fillchar = '0', side = 'left') + df['DATE'].dt.minute.astype('str').str.pad(width = 2, fillchar = '0') + ":" + 
  df[df.columns[2]].str.replace(" ","").str.split(':').str.get(0) + ":" + xlsEnd) 
df['Status'] = 'OKAY' # set status for the new playlist list as OKAY -- used in dupCheck comaprison
fields = ['ID', 'Date', 'SchedTime', 'EngagementActivity/Notes', 'WorkGroup', 'Terminal', 'Status'] ## fields list created to ensure correct order and columns

# checks if the playlist refrence file exists and saves a new one if not
if os.path.isfile('PlaylistRef.csv') == False:
    print('Saving Initial Playlist Reference...')
    df[fields].to_csv('PlaylistRef.csv',header = True, index = False)
    print('Initial Playlist Saved!')
    
elif os.path.isfile('PlaylistRef.csv') == True:
    from dupCheck import dupCheck ## the dupCheck script needs to be in the same working directory that you set in line 9
    
    check_df = pd.read_csv(r'.\PlaylistRef.csv') # loads the playlist reference for comparison
    status_change = dupCheck(df[fields], check_df) # passes the new playlist and the reference playlist to an external function
    
    # concatentating the dataframes with only the columns we want -- matching the status change output
    combined_df = pd.concat([check_df.loc[:, 'ID':'Status'], status_change], sort = True).reset_index(drop = True) 
    combined_df['SchedTime'] = pd.to_datetime(combined_df['SchedTime'], format = "%H:%M:%S").dt.time ## changing the schedtime to datetime format so we can sort the dataframe
    # there are going to be some duplicated unique ids since we are simply concatenating the status_change rows with the new playlist list
    ## First, getting the unique IDs that have be duplicated
    dup_id = combined_df[combined_df.duplicated(['ID'])]['ID'].tolist()
    # subset the dataframe from those duplicated IDs and only get the rows that still have the 'OKAY' status
    ## we grab the index of the those rows, then drop them from our cleaned new playlist item
    dup_idx = combined_df.loc[(combined_df['ID'].isin(dup_id)) & (combined_df['Status'] == 'OKAY')].index
    combined_df = combined_df.drop(combined_df.index[dup_idx])
    # We save the new playlist with the flags over the playlist reference file and the whole process repeats
    print('Saving Playlist Reference...')
    ## can be pointed to sql database but requires pyodbc and credentials
    combined_df[fields].sort_values(by = 'SchedTime').to_csv('PlaylistRef.csv', header = True, index = False) 
    print('Playlist Saved!')
#end = time.time()
#print('Runtime:', round(end - start, 5), 'seconds') ## runtime is ~13 seconds
