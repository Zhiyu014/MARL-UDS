# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 17:26:56 2021

@author: chong
"""

import pandas as pd
from datetime import datetime

rainfiles = pd.read_csv('./rainfiles.csv',index_col=0)
rainfiles['tot'] = rainfiles.sum(axis=1)
rainfiles['datetime'] = rainfiles['date']+' '+rainfiles['time']
rains = rainfiles.drop(rainfiles[rainfiles['tot']==0].index)
rainids = rains.index.to_list()


ranges =[]


import numpy as np
diff = np.diff(rainids)
rang = [rainids[0]]
for k,i in enumerate(diff):
    if i>=24:
        ranges.append(rang)
        rang = [rainids[k+1]]
    else:
        rang.append(rainids[k+1])
        
ranges = [rain for rain in ranges if len(rain)>=12]
ranges = [(min(rang),max(rang)) for rang in ranges]




precips = [rainfiles.loc[[i for i in range(*rang)],'tot'].sum() for rang in ranges]
rain_samples = pd.DataFrame(columns = ['Start','End','Precip'])
rain_samples['Start'] = rainfiles.loc[[a for a,_ in ranges],'datetime'].to_list()
rain_samples['End'] = rainfiles.loc[[b for _,b in ranges],'datetime'].to_list()
rain_samples['Precip'] = precips
rain_samples['date'] = rain_samples['Start'].apply(lambda st:st[:10])
rain_samples['duration'] = rain_samples['End'].apply(lambda st:datetime.strptime(st,'%m/%d/%Y %H:%M:%S'))-rain_samples['Start'].apply(lambda st:datetime.strptime(st,'%m/%d/%Y %H:%M:%S'))
rain_samples['duration'] = rain_samples['duration'].apply(lambda du:du.total_seconds()/300)

rain_samples['dry_time'] = rain_samples.apply(lambda row:(rainfiles.loc[row['Start']:row['End'],'tot']==0).sum()/row['duration'],axis=1)
rain_samples.to_csv('./rain_2h.csv')








