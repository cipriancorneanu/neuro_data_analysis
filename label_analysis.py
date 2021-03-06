__author__ = 'cipriancorneanu'

from xml.dom import minidom
import matplotlib.pylab as plt
import numpy as np
import os
import math

def load_data(filename):
    xmldoc = minidom.parse(filename)
    labels = xmldoc.getElementsByTagName("Label")

    data = []
    for label in labels:
        id = label.getElementsByTagName("IDType")[0].childNodes[0].data
        start = label.getElementsByTagName("Start")[0].childNodes[0].data
        end = label.getElementsByTagName("End")[0].childNodes[0].data

        data.append([int(id), int(start), int(end)])

    return convert_data(data)

def load_names(filename):
    xmldoc = minidom.parse(filename)
    labels = xmldoc.getElementsByTagName("Type")

    names = ['NO LABEL']
    for label in labels:
        id = label.getElementsByTagName("ID")[0].childNodes[0].data
        name = label.getElementsByTagName("Name")[0].childNodes[0].data
        names.append(str(name))
    return names

def convert_data(x):
    t_scale = 1000
    max_range = int(x[-1][2]/t_scale)
    out = np.zeros(max_range)

    for label in x:
        out[int(label[1]/t_scale):int(label[2]/t_scale)] = label[0]+1

    return out

def plot_data(x1, x2, y, id):
    # Two subplots, the axes array is 1-d
    RANGE  = 120 # 2 mins

    #for i in range(1,int(len(x1)/RANGE)):
    plt.figure()
    x = range(0,len(x1))
    plt.plot(x, x1[x], color = 'blue', linewidth=2.0)
    plt.plot(x, x2[x], 'r--', linewidth = 2.0)

    plt.xlabel(r'Time(secs)', fontsize=20)
    plt.ylabel(r'Emotion', fontsize=20)
    y.append('')
    plt.yticks(range(0,12), y, rotation = 15, fontsize = 8)
    plt.legend(['First labeler', 'Second labeler'], loc = 'upper left')
    plt.grid()
    plt.title(id)
    plt.savefig(id+'_dt.eps')

def equalize(x1, x2):
    #if not equal length add zeros
    diff = len(x1) - len(x2)
    if diff>0:
        x2 = np.append(x2,np.zeros(diff))
    else:
        x1 = np.append(x1,np.zeros(abs(diff)))

    return (x1,x2)

def stats(x1,x2):
    return np.corrcoef(x1,x2)

def confusion_mat(x1,x2):
    N = 11
    m = np.zeros([N,N])
    for i in range(0,len(x1)):
        m[x1[i]][x2[i]] += 1

    m_normalized = m.astype('float')

    return m_normalized

def plot_confusion_matrix(cm, y, id, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(m))
    plt.xticks(tick_marks, y, rotation='vertical')
    plt.yticks(tick_marks, y)
    plt.xlabel('Labeler 2', fontsize = 20)
    plt.ylabel('Labeler 1', fontsize = 20)
    plt.title(id)
    plt.savefig(id+'_cmat.png')


if __name__ == "__main__":
    ipath = '/Users/cipriancorneanu/Research/data/neurochild/data/labels/' # Add your path here
    filename_labels = 'label_names.xml'

    # Load label codes
    names = load_names(filename_labels)

    # Read label files
    fnames = [f for f in os.listdir(ipath) if f.endswith('.xml')]

    # Split by labelers
    fnames_1 = [f for f in fnames if f.split('_')[1].split('.')[0] == '1']
    fnames_2 = [f for f in fnames if f.split('_')[1].split('.')[0] == '2']

    # Read
    data1 = [load_data(ipath+f) for f in fnames_1]
    data2 = [load_data(ipath+f) for f in fnames_2]

    # Equalize
    data = [equalize(dt1, dt2) for dt1, dt2 in zip(data1, data2)]

    # Split
    data1 = np.concatenate([d[0] for d in data])
    data2 = np.concatenate([d[1] for d in data])

    # Plot correlation btw labelers
    print stats(data1, data2)

    # Plot labels
    plot_data(data1, data2, names, '')

    # Plot confusion mat
    m = confusion_mat(data1, data2)
    plot_confusion_matrix(m, names, 'video_'+str(id))

    # Plot label intersection distribution
    x, bins = np.histogram([x for x, y in zip(data1, data2) if x==y], bins = 11, range = (-0.5, 10.5))
    x = [float(item)/np.sum(x) for item in x]

    ind_sort = np.argsort(x)
    for i in ind_sort:
        print '{} : {}'.format(names[i], x[i])
