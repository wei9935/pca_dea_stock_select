import pandas as pd
import gurobipy
from gurobipy import quicksum
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# PCA model with sklearn
class pca_IO(object):
    def __init__(self, data, in_list, out_list):
        self.in_df = data.loc[:,in_list]
        self.out_df = data.loc[:,out_list]

    def pca_calculate(self, df, var_level=0.85):
        scaler = StandardScaler()
        sd = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        pca = PCA()
        pca.fit(sd)
        
        if var_level == 'kaiser':# Kaiser Criterion
            num_components = sum(pca.explained_variance_ > 1) if var_level == 'kaiser' else num_components
        else: # Explained Variance Threshold
            var = pca.explained_variance_ratio_.cumsum()
            num_components = (var >= var_level).argmax() + 1
        
        pca_df = pca.transform(df)[:, :num_components]
        pca_df = pd.DataFrame(pca_df, columns=[f'PC{i+1}' for i in range(num_components)])
        
        #result = pca_df.transform(sd)
        return pca_df
    
    def fit(self, var_level=0.85):
        in_pca = self.pca_calculate(self.in_df, var_level=var_level)
        out_pca = self.pca_calculate(self.out_df, var_level=var_level)
        
        scaler = MinMaxScaler()
        in_pca = pd.DataFrame(scaler.fit_transform(in_pca), columns=in_pca.columns, index=in_pca.index)
        out_pca = pd.DataFrame(scaler.fit_transform(out_pca), columns=out_pca.columns, index=out_pca.index)
        return in_pca, out_pca
    
# DEA CCR-model
class DEA(object):
    def __init__(self, DMUs_Name, X, Y, AP=False):
        self.m1, self.m1_name = X.shape[1], X.columns.tolist()
        self.m2, self.m2_name = Y.shape[1], Y.columns.tolist()
        self.AP = AP
        self.DMUs, self.X, self.Y = gurobipy.multidict({DMU: [X.loc[DMU].tolist(),
                                                              Y.loc[DMU].tolist()] for DMU in DMUs_Name})
        print(f'DEA(AP={AP}) MODEL RUNING...')

    def __CCR(self): ##input-oriented
        for k in self.DMUs:
            MODEL = gurobipy.Model()

            OE = MODEL.addVar()
            lambdas = MODEL.addVars(self.DMUs)

            MODEL.update()
            ## update environment
            MODEL.setObjective(OE, sense=gurobipy.GRB.MINIMIZE)

            MODEL.addConstrs(quicksum(lambdas[i] * self.X[i][j] for i in self.DMUs) <= OE * self.X[k][j] for j in range(self.m1))
            MODEL.addConstrs(quicksum(lambdas[i] * self.Y[i][j] for i in self.DMUs) >= self.Y[k][j] for j in range(self.m2))
            MODEL.setParam('OutputFlag', 0)
            MODEL.setParam('NonConvex',2)

            MODEL.optimize()

            self.Result.at[k, ('Efficiency Analysis', 'CCR')] = MODEL.objVal

        return self.Result

    def dea(self):
        columns_Page = ['Efficiency Analysis']
        columns_Group = ['CCR']
        self.Result = pd.DataFrame(index=self.DMUs, columns=[columns_Page, columns_Group])
        self.__CCR()
        self.Result.columns = ['Efficiency']

        return self.Result

    def analysis_to_excel(self, file_name=None):
        Result = self.dea()
        file_name = 'DEA_report.xlsx' if file_name is None else f'\\{file_name}.xlsx'
        Result.to_excel(file_name, 'DEA_report')
    
    def dmus_efficiency(self, top_n=None):
        Result = self.Result
        self.max_efficiency = round(Result['Efficiency'].max())
        self.efficient_dmus = Result[Result['Efficiency']==self.max_efficiency].dropna().index.to_list()
        self.inefficient_dmus = Result[Result['Efficiency']!=self.max_efficiency].dropna().index.to_list()
        
        if top_n == None:
            top_n = len(self.efficient_dmus)
        Result_val = pd.Series(Result['Efficiency'], index=Result.index)
        sorted_indices = sorted(range(len(Result_val)), key=lambda i: Result_val[i])
        self.n_efficient_dmus = [Result.index[i] for i in sorted_indices[:top_n]]
        self.n_inefficient_dmus = [Result.index[i] for i in sorted_indices[-top_n:]]
