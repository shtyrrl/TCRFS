#coding=UTF-8
import numpy as np
import skfeature.utility.entropy_estimators as ees

def TCRFS(data,labels,n_features):
    """
    fi=candidate feature  Y=[l1,l2,..lm] f_selected=S(selected subset)
    J=sum(I(fi;lj)+sum[sum(I(f_selected;lj|fi))-sum[sum(I(fi;f_selected;lj))]
    :param data:  numpy.array [n_sample,n_feature]
    :param labels: numpy.array [n_sample,n_classes]
    :param n_features: the number of selected features
    :return:F list
    """
    F=[]
    mm,f_nub=np.shape(data)
    nn,l_nub=np.shape(labels)
    nl=1/float(l_nub)
    nl2=1/float(l_nub-1)
    rel=np.zeros((f_nub,1))
    for i in range(f_nub):
        for j in range(l_nub):
            for k in range(l_nub):
                if k!=j:
                    rel[i]+= ees.cmidd(data[:,i],labels[:,j],labels[:,k])

    #Selecting First Feature
    idx=np.argmax(rel[:,0])
    F.append(idx)
    f_select=data[:,idx]
    ffmi1=np.zeros((f_nub,1))
    fcmi =np.zeros((f_nub,1))
    redu=np.zeros((f_nub,1))
    while len(F)<n_features:
        j_cmi=-10000000000000
        ns=1/float(len(F))
        for i in range(f_nub):
            if i not in F:
                for ii in range(l_nub):
                    fcmi[i]+=ees.cmidd(data[:,i],labels[:,ii],f_select)+ees.cmidd(f_select,labels[:,ii],data[:,i])
                    redu[i]+=ees.cmidd(data[:,i],f_select,labels[:,ii])
                ffmi1[i] += ees.midd(data[:,i],f_select)

                t=nl*nl2*rel[i]+ns*nl*fcmi[i]-(ns*ffmi1[i]-ns*nl*redu[i])
                if t > j_cmi:
                    j_cmi = t
                    idx = i
        F.append(idx)
        f_select=data[:,idx]
    return F

print (TCRFS(X,y,20))
