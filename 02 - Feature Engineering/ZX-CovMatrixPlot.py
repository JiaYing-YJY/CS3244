##############################
#  MISSING CODE HERE         #
#  EG. REQUIRED MODULES      #
#      DATA IMPORT AND CLEAN #
##############################

import seaborn as sns
cols = list(df.columns)
sns.pairplot(df[cols], size=2.0)

from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 
import seaborn as sns
cols = list(df.columns)
stdsc = StandardScaler() 
X_std = stdsc.fit_transform(df[cols].iloc[:,:].values)
cov_mat =np.cov(X_std.T)
plt.figure(figsize=(50,50))
sns.set(font_scale=3)
hm = sns.heatmap(cov_mat,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 20},
                 cmap='coolwarm',                 
                 yticklabels=cols,
                 xticklabels=cols)
plt.title('Covariance matrix showing correlation coefficients', size = 40)
plt.tight_layout()
#plt.show()
if True: plt.savefig("CovMatrixPlot.png",dpi=300)