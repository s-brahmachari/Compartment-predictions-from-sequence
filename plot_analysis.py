import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.pyplot import cm 

hyper_param = "3feature"
# with open("./analysis_data/neighbor_importance.pickle", "rb") as fr:
# with open(f"./analysis_data/hyper_param_search_{hyper_param}.pickle", "rb") as fr:
with open("./analysis_data/feature_importance_use3feature.pickle", "rb") as fr:
    data = pickle.load(fr)

fig,ax=plt.subplots(2,2, figsize=(10,8), dpi=300, 
                    gridspec_kw={"hspace":0.3, "wspace":0.3})
colors = cm.rainbow(np.linspace(0,1,30))
count=0
for val in data['replica1']:
    if val=="0.05": continue
    ax[0][0].plot(data['replica1'][val]["train_loss"], '--',c=colors[count],)
    ax[0][0].plot(data['replica1'][val]["val_loss"], c=colors[count],label=val)
    ax[0][1].plot(data['replica1'][val]["train_accu"], '--', c=colors[count], )
    ax[0][1].plot(data['replica1'][val]["val_accu"], c=colors[count],label=val)
    count+=1

roc = {}
test_accu = {}

for rep in data:
    if 'hyper' in rep: continue
    for val in data[rep].keys():
        if val=="0.05": continue
        # print(data[rep][val].keys())
        if val in roc:
            roc[val].append(data[rep][val]["roc"])
        else:
            roc[val]=[data[rep][val]["roc"]]

        if val in test_accu:
            test_accu[val].append(data[rep][val]["test_accu"])
        else:
            test_accu[val]=[data[rep][val]["test_accu"]]

# param=[]        
# for val in roc_dct:
#     param.append(val)
#     roc_av.append(np.mean(roc_dct[val]))
#     roc_std.append(np.std(roc_dct[val]))

# print(len(roc_av), len(param))

ax[0][0].set_xlabel("epochs", fontsize=14)
ax[0][1].set_xlabel("epochs", fontsize=14)
ax[0][0].set_ylabel("Loss", fontsize=14)
ax[0][1].set_ylabel("Accuracy", fontsize=14)

labels=sorted(roc.keys(), key=lambda x: x[1])
labels_plot = [xx if len(xx.split("-"))<5 else "All" for xx in labels]

# labels_plot=[]

# for xx in labels:
#     if len(xx.split("-"))==6: 
#         all_features=xx

# for xx in labels:    
#     if len(xx.split("-"))==6: labels_plot.append("All")
#     else:
#         name=[]
#         for yy in all_features.split("-"):
#             if yy not in xx:
#                 name.append(yy)
#         labels_plot.append("$\\Delta$ "+"-".join(name))


ax[1][0].boxplot([roc[xx] for xx in labels], 
                    labels=labels)

ax[1][1].boxplot([test_accu[xx] for xx in labels], 
                    labels=labels)

ax[1][0].set_ylabel("AUC-ROC", fontsize=14)
ax[1][1].set_ylabel("Test accuracy", fontsize=14)

print(ax[1][1].get_xticks())
for axi in ax[1]:
    axi.set_xticklabels([xx for xx in labels_plot], rotation = 90)
    # axi.set_xlabel("Number of neighbors in input", fontsize=14)
# print(param, roc_av, roc_std)   
# ax[0][0].legend()
# ax[0][1].legend() 
# plt.plot(data['replica4']['4']["train_loss"])
fig.savefig("test.png",bbox_inches="tight")
fig.savefig(f"./analysis_data/plots/{hyper_param}.png", dpi=300, bbox_inches="tight")