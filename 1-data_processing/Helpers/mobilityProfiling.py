import os
import math
import argparse
import numpy as np
import pandas as pd
from csv import writer
import statistics as stat
from sklearn import mixture

#Function to append list of returns or explorations to a CSV.
def appendListToCSV (file, listElements):

	with open(file, 'a+', newline = '') as write_obj:
		csv_writer = writer(write_obj)
		csv_writer.writerow(listElements)

def mobilityProfilingFeatures (dataset, listUsr):

	known_places = dict()

	for usr in listUsr:

		tab_vis = []
		fr = dataset.loc[[usr]]
		fr = fr.sort_values(by = 'date') 
		dist_freq = fr[['geohash','date']].groupby('geohash').count().date
		dist_freq_val = dist_freq.values

		for i in range(len(dist_freq_val)):

		    if dist_freq_val[i] > (max(dist_freq_val) * 0.9):
		        tab_vis.append(dist_freq.index[i])
		    
		known_places[usr] = tab_vis

	features = dict()

	nb_measure = []

	features['nb_succ_u_m'] = []
	features['nb_succ_r_m'] = []

	for k in listUsr:

	        #Distributions
	        dist_exp = []
	        dist_ret = []
	        dist_ret_dist = []
	        dist_w_t = []
	        
	        tab_visit = known_places.get(k)

	        cmp_exp = 0
	        cmp_ret = 0
	        cmp_ret_dist = 0
	        
	        cmp_day = 0
	        
	        cmp_measure = 0
	        fr = dataset.loc[[k]]
	        fr = fr.sort_values(by = 'date')    
	        fr = fr.set_index(['date','geohash'])
	        dd = 0

	        for i in fr.index:
	            
	            if cmp_day <= 10:  
	                cmp_measure = cmp_measure+1
	                
	                if cmp_exp == 0 and cmp_ret == 0:

	                    prev_visit = i[1]
	                    cmp_day = 1
	                    prev_day = i[0].date()
	                    last = i[0]
	                    cmp_exp = cmp_exp+1
	                    tab_visit.append(i[1])
	                    dd = 1
	                    
	                elif (i[0].date()-prev_day).days > 1:

	                        if dd == 1:
	                            cmp_day = cmp_day+1
	                            dd = 0

	                        prev_day = i[0].date()

	                        if cmp_exp>0:
	                            dist_exp.append(cmp_exp)  

	                        if cmp_ret>0:
	                            dist_ret.append(cmp_ret)
	                            dist_ret_dist.append(cmp_ret_dist)
	                            dd = 1

	                        cmp_exp = 0
	                        cmp_ret = 0 
	                        cmp_ret_dist = 0
	                        dist_w_t.append((24-(last.hour))*60)
	                            
	                        if i[1] in tab_visit:  
	                            cmp_ret_dist = cmp_ret_dist+1 
	                            cmp_ret = cmp_ret+1
	                            dd = 1

	                        else:
	                            cmp_exp = cmp_exp+1
	                            tab_visit.append(i[1])
	  
	                        last = i[0]
	    
	                else:
	                    #if known place
	                    if i[1] in tab_visit:

	                        if cmp_ret == 0:

	                            if cmp_exp>0:

	                                dist_exp.append(cmp_exp)
	                                cmp_exp = 0
	                                dist_w_t.append(((i[0])-(last)).total_seconds()/3600)
	                                last = i[0]
	                                dd = 1

	                            cmp_ret_dist = cmp_ret_dist+1

	                        elif i[1] !=  prev_visit:

	                            cmp_ret_dist = cmp_ret_dist+1
	                            dist_w_t.append(((i[0])-(last)).total_seconds()/3600)
	                            last = i[0]

	                        cmp_ret = cmp_ret+1

	                    else:

	                        if cmp_exp == 0:

	                            if cmp_ret>0:

	                                dist_ret.append(cmp_ret)
	                                dist_ret_dist.append(cmp_ret_dist)
	                                cmp_ret = 0
	                                cmp_ret_dist = 0 
	                                dist_w_t.append(((i[0])-(last)).total_seconds()/3600)
	                                last = i[0]
	                                dd = 1

	                        cmp_exp = cmp_exp+1
	                        tab_visit.append(i[1])
	                        
	            if i[0].date().day!= prev_day.day:

	                if dd == 1:
	                    cmp_day = cmp_day+1
	                    
	                dd = 0

	            prev_visit = i[1]
	        
	        nb_measure.append(cmp_measure)
	        
	        if cmp_ret>0: 
	            dist_ret.append(cmp_ret)
	            dist_ret_dist.append(cmp_ret_dist)
	            
	        if cmp_exp>0:    
	            dist_exp.append(cmp_exp)

	        m_exp = stat.mean(dist_exp)

	        if len(dist_exp)>1:
	            sig_exp = stat.stdev(dist_exp, m_exp)
	            cv_exp = (sig_exp/m_exp)*100

	        else:
	            cv_exp = 0

	        if len(dist_ret)>0:
	            m_ret = stat.mean(dist_ret)

	        else:
	            m_ret = 0

	        if len(dist_ret)>1:
	            sig_ret = stat.stdev(dist_ret, m_ret)
	            cv_ret = (sig_ret/m_ret)*100

	        else:
	            cv_ret = 0

	        if len(dist_ret_dist)>0:
	            m_ret_dist = stat.mean(dist_ret_dist)

	        else:
	            m_ret_dist = 0

	        if len(dist_ret_dist)>1:
	            sig_ret_dist = stat.stdev(dist_ret_dist, m_ret_dist)

	            if m_ret_dist>0:
	                cv_ret_dist = (sig_ret_dist/m_ret_dist)*100

	            else:
	                cv_ret_dist = 0

	        else:
	            cv_ret_dist = 0

	        features.get('nb_succ_u_m').append((m_exp))
	        features.get('nb_succ_r_m').append((m_ret))

	features['inter'] = []
	features['deg_ret'] = []

	for i in range(0, len(features.get('nb_succ_u_m'))):
		features.get('inter').append((features.get('nb_succ_u_m')[i]+features.get('nb_succ_r_m')[i]))
		features.get('deg_ret').append((math.degrees(np.arctan(features.get('nb_succ_r_m')[i]/features.get('nb_succ_u_m')[i]))))

	return features

	

def mobilityProfiling (dataset, listUsr):

	known_places = dict()

	for usr in listUsr:

		tab_vis = []
		fr = dataset.loc[[usr]]
		fr = fr.sort_values(by = 'date') 
		dist_freq = fr[['geohash','date']].groupby('geohash').count().date
		dist_freq_val = dist_freq.values

		for i in range(len(dist_freq_val)):

		    if dist_freq_val[i] > (max(dist_freq_val) * 0.9):
		        tab_vis.append(dist_freq.index[i])
		    
		known_places[usr] = tab_vis

	features = dict()

	nb_measure = []

	features['nb_succ_u_m'] = []
	features['nb_succ_r_m'] = []

	for k in listUsr:

	        #Distributions
	        dist_exp = []
	        dist_ret = []
	        dist_ret_dist = []
	        dist_w_t = []
	        
	        tab_visit = known_places.get(k)

	        cmp_exp = 0
	        cmp_ret = 0
	        cmp_ret_dist = 0
	        
	        cmp_day = 0
	        
	        cmp_measure = 0
	        fr = dataset.loc[[k]]
	        fr = fr.sort_values(by = 'date')    
	        fr = fr.set_index(['date','geohash'])
	        dd = 0

	        for i in fr.index:
	            
	            if cmp_day <= 10:  
	                cmp_measure = cmp_measure+1
	                
	                if cmp_exp == 0 and cmp_ret == 0:

	                    prev_visit = i[1]
	                    cmp_day = 1
	                    prev_day = i[0].date()
	                    last = i[0]
	                    cmp_exp = cmp_exp+1
	                    tab_visit.append(i[1])
	                    dd = 1
	                    
	                elif (i[0].date()-prev_day).days > 1:

	                        if dd == 1:
	                            cmp_day = cmp_day+1
	                            dd = 0

	                        prev_day = i[0].date()

	                        if cmp_exp>0:
	                            dist_exp.append(cmp_exp)  

	                        if cmp_ret>0:
	                            dist_ret.append(cmp_ret)
	                            dist_ret_dist.append(cmp_ret_dist)
	                            dd = 1

	                        cmp_exp = 0
	                        cmp_ret = 0 
	                        cmp_ret_dist = 0
	                        dist_w_t.append((24-(last.hour))*60)
	                            
	                        if i[1] in tab_visit:  
	                            cmp_ret_dist = cmp_ret_dist+1 
	                            cmp_ret = cmp_ret+1
	                            dd = 1

	                        else:
	                            cmp_exp = cmp_exp+1
	                            tab_visit.append(i[1])
	  
	                        last = i[0]
	    
	                else:
	                    #if known place
	                    if i[1] in tab_visit:

	                        if cmp_ret == 0:

	                            if cmp_exp>0:

	                                dist_exp.append(cmp_exp)
	                                cmp_exp = 0
	                                dist_w_t.append(((i[0])-(last)).total_seconds()/3600)
	                                last = i[0]
	                                dd = 1

	                            cmp_ret_dist = cmp_ret_dist+1

	                        elif i[1] !=  prev_visit:

	                            cmp_ret_dist = cmp_ret_dist+1
	                            dist_w_t.append(((i[0])-(last)).total_seconds()/3600)
	                            last = i[0]

	                        cmp_ret = cmp_ret+1

	                    else:

	                        if cmp_exp == 0:

	                            if cmp_ret>0:

	                                dist_ret.append(cmp_ret)
	                                dist_ret_dist.append(cmp_ret_dist)
	                                cmp_ret = 0
	                                cmp_ret_dist = 0 
	                                dist_w_t.append(((i[0])-(last)).total_seconds()/3600)
	                                last = i[0]
	                                dd = 1

	                        cmp_exp = cmp_exp+1
	                        tab_visit.append(i[1])
	                        
	            if i[0].date().day!= prev_day.day:

	                if dd == 1:
	                    cmp_day = cmp_day+1
	                    
	                dd = 0

	            prev_visit = i[1]
	        
	        nb_measure.append(cmp_measure)
	        
	        if cmp_ret>0: 
	            dist_ret.append(cmp_ret)
	            dist_ret_dist.append(cmp_ret_dist)
	            
	        if cmp_exp>0:    
	            dist_exp.append(cmp_exp)

	        m_exp = stat.mean(dist_exp)

	        if len(dist_exp)>1:
	            sig_exp = stat.stdev(dist_exp, m_exp)
	            cv_exp = (sig_exp/m_exp)*100

	        else:
	            cv_exp = 0

	        if len(dist_ret)>0:
	            m_ret = stat.mean(dist_ret)

	        else:
	            m_ret = 0

	        if len(dist_ret)>1:
	            sig_ret = stat.stdev(dist_ret, m_ret)
	            cv_ret = (sig_ret/m_ret)*100

	        else:
	            cv_ret = 0

	        if len(dist_ret_dist)>0:
	            m_ret_dist = stat.mean(dist_ret_dist)

	        else:
	            m_ret_dist = 0

	        if len(dist_ret_dist)>1:
	            sig_ret_dist = stat.stdev(dist_ret_dist, m_ret_dist)

	            if m_ret_dist>0:
	                cv_ret_dist = (sig_ret_dist/m_ret_dist)*100

	            else:
	                cv_ret_dist = 0

	        else:
	            cv_ret_dist = 0

	        features.get('nb_succ_u_m').append((m_exp))
	        features.get('nb_succ_r_m').append((m_ret))

	features['inter'] = []
	features['deg_ret'] = []

	for i in range(0, len(features.get('nb_succ_u_m'))):
		features.get('inter').append((features.get('nb_succ_u_m')[i]+features.get('nb_succ_r_m')[i]))
		features.get('deg_ret').append((math.degrees(np.arctan(features.get('nb_succ_r_m')[i]/features.get('nb_succ_u_m')[i]))))

	scouters = []
	routiners = []
	regulars = []

	scoutersContents = ['scouters (dev_id)']
	routinersContents = ['routiners (dev_id)']
	regularsContents = ['regulars (dev_id)']

	appendListToCSV('scouters.csv', scoutersContents)
	appendListToCSV('routiners.csv', routinersContents)
	appendListToCSV('regulars.csv', regularsContents)

	nb_clusters = 3

	gmm = mixture.GaussianMixture(covariance_type="full", n_components=nb_clusters).fit(np.array([features.get('deg_ret'),features.get('inter')]).T)
	labels_nb = gmm.predict(np.array([ features.get('deg_ret'),features.get('inter')]).T)

	exp = min(features.get('deg_ret'))
	ret = max(features.get('deg_ret'))

	for i in range(0, len(labels_nb)):

	    if features.get('deg_ret')[i] == exp:
	        explorers=labels_nb[i]

	    elif features.get('deg_ret')[i] == ret:
	        returners=labels_nb[i]

	for i in range(0, len(labels_nb)):

	    if labels_nb[i] == explorers:
	        scouters.append(listUsr[i])

	    elif labels_nb[i] == returners:
	        routiners.append(listUsr[i])

	    else:
	        regulars.append(listUsr[i])

	appendListToCSV('scouters.csv', scouters)
	appendListToCSV('routiners.csv', routiners)
	appendListToCSV('regulars.csv', regulars)

def createDataset (filttraj):

	initialDataset = pd.read_csv(filttraj)
	initialDataset['date'] = pd.to_datetime(initialDataset['date'])
	datasetIndex = initialDataset.set_index('dev_id')
	listUsr = datasetIndex.index
	listUsr = listUsr.drop_duplicates()

	mobilityProfiling (initialDataset, listUsr)

def main ():

	parser = argparse.ArgumentParser()
	parser.add_argument("-dt", "--filttraj", help = "Dataset with pruned trajectories")
	args = parser.parse_args()

	if(os.path.isfile('regulars.csv') or os.path.isfile('routiners.csv') or os.path.isfile('scouters.csv')):
		print ("File regulars.csv and/or routiners.csv and/or scouters.csv already exists, delete it before running the mobility profiling module.")
		exit()

	createDataset (args.filttraj)

if __name__ ==  '__main__':
	main()