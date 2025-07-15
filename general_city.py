#%config Completer.use_jedi = False
import math
import pandas as pd 
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.decomposition import PCA
import holidays 
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import plotly.express as px 
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

class Feature_Extraction():
    
    def GET_deleted_columns(self,data,variables_to_remove):
        """
        This functions is used to maintain the list of variables which are highly collinear with the 
        output variable and must be removed to maintain a good list of training data
        """
        for variable in variables_to_remove:
            try:
                data = data.drop(columns = [variable])
            except KeyError:
                pass
        return data

    def GET_Scaled_data(self,data):
        """
        Scaling the data so that all variables are in the same range.
        """

        column_names = data.columns
        scaler = StandardScaler()
        #print(scaler.fit(data))
        scaled_data = scaler.fit_transform(data)
        scaled_data = pd.DataFrame(scaled_data)
        scaled_data.columns = column_names
        return scaled_data, scaler
    
    def GET_data_splits(self,X,Y):
        """
        The objective of this function is to split the data into it's own train/test/validation splits
        X_train, X_test, y_train, y_test
        """
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, shuffle = True)
        
        X_train = X_train.reset_index(inplace = False)
        X_train = X_train.drop(columns = ["index"])
        
        X_test = X_test.reset_index(inplace = False)
        X_test = X_test.drop(columns = ["index"])
        
        y_train = y_train.reset_index(inplace = False)
        y_train = y_train.drop(columns = ["index"])
        
        y_test = y_test.reset_index(inplace = False)
        y_test = y_test.drop(columns = ["index"])
        return X_train, X_test, y_train, y_test
    
    def GET_reduce_time_features(self,df,variable_name):
        df[variable_name] = pd.to_datetime(df[variable_name],unit='s')
        list_of_times = []
        for i in range(0,len(df)):
            date = str(df[variable_name][i])
            datem = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
            extracted_time = [datem.year,datem.month,datem.day,datem.hour]
            list_of_times.append(extracted_time)
        combined = pd.DataFrame(list_of_times)
        combined.columns = [variable_name+"_year",
                           variable_name+"_month",
                           variable_name+"_day",
                           variable_name+"_hour"]
        merged_df = pd.concat([df, combined], axis=1, join='inner')
        merged_df = merged_df.drop(columns = [variable_name])
        return merged_df
    
    def GET_time_features(self,data):
        data = self.SET_day_type(data,"start_time")
        data = self.GET_reduce_time_features(data,"start_time")
        data = self.GET_reduce_time_features(data,"end_time")
        return data 
    
    def SET_day_type(self,data,variable_name):
        # Weekdays = 0 
        # Weekend = 1
        # Holiday = 2
        data["temp"] = pd.to_datetime(data[variable_name],unit='s')
        result = []
        import holidays
        us_holidays = holidays.US()
        for i in range(0,len(data)):
            current_time = data["temp"][i]
            if current_time in us_holidays:
                result.append(2)
            else:
                day_type = current_time.weekday()
                if day_type <= 5:
                    result.append(0)
                if day_type >5:
                    result.append(1)
        data["day_type"] = result
        data = data.drop(columns = ["temp"])
        return data
    
    def GET_X_and_Y(self,data,target_variable):
        X = data[[item for item in data.columns if item != target_variable]]
        Y = data[[target_variable]]
        return X, Y
    
    def GET_BUS_DESIGN(self,data):    
        manufacturer = []
        capacity = []
        missing = 0 
        for i in range(0,len(data)):
            current = data["bus_id"][i]
            #print("fault")
            # 1 - Corresponds to New Flyer -> culver city 
            # 2 - Corresponds to Portera  - > foothill 
            # 3 - Corresponds to Alexander Dennis
            if str(current)[0:1] == "7":
                manufacturer.append(1)
                if current == 7157:
                    capacity.append(350)
                else:
                    # this includes 7156, 7158, 7159
                    capacity.append(440)
            if str(current)[0:1] in "2":
                manufacturer.append(2)
                if current in list(range(2001,2016)):
                    capacity.append(72)
                if current in list(range(2016,2018)):
                    capacity.append(79)
                if current in list(range(2600,2614)):
                    capacity.append(440)
                if current in list(range(2800,2803)):
                    capacity.append(440)

            if str(current)[0:2] == "30":
                manufacturer.append(3)
                if current in [3000,3001]:
                    capacity.append(648)

        if len(capacity) != len(data) or  len(manufacturer) != len(data) :
            print("UNIDENTIFIED BUS_ID ENTERED")

        else:
            data["BATTERY_CAPACITY"] = capacity
            data['MANUFACTURER'] = manufacturer
            data = data.drop(columns = ["bus_id"])
            return data
        
class CLUSTERING_MODEL(): 
    def __init__(self,num_iterations,num_clusters,data):
        self.iterations = num_iterations
        self.clusters = num_clusters
        self.data = data.copy()

    def gmm(self):
        from sklearn.mixture import GaussianMixture
        total_predictons = []
        for i in range(0,self.iterations):
            gm = GaussianMixture(n_components=self.clusters , random_state=0).fit_predict(self.data)
            total_predictons.append(pd.DataFrame(gm).T)
        concatenated = pd.concat(total_predictons)
        round_kmeans_predictions = round(concatenated.mean(axis=0))
        self.data["GMM_Prediction"]=round_kmeans_predictions
        self.data['GMM_Prediction']=self.data['GMM_Prediction'].replace(0,-1)

    def agglomerative(self):
        num_iterations = self.iterations/10
        from sklearn.cluster import AgglomerativeClustering
        total_predictons = []
        for i in range(0,num_iterations):
            prediction = AgglomerativeClustering().fit_predict(self.data)
            total_predictons.append(pd.DataFrame(prediction).T)
        concatenated = pd.concat(total_predictons)
        agglomerative_predictions = round(concatenated.mean(axis=0))
        self.data["Agglomerative"]=agglomerative_predictions
        self.data['Agglomerative']=self.data['Agglomerative'].replace(0,-1)
        return data

    def kmeans(self):
        num_iterations = self.iterations
        from sklearn.cluster import KMeans
        total_predictons = []
        for i in range(0,num_iterations):
            kmeans = KMeans(n_clusters=self.clusters, random_state=0).fit(self.data)
            current_kmeans_predictions = kmeans.labels_
            total_predictons.append(pd.DataFrame(current_kmeans_predictions).T)
        concatenated = pd.concat(total_predictons)
        round_kmeans_predictions = round(concatenated.mean(axis=0))
        self.data["K_Means_Prediction"]=round_kmeans_predictions
        self.data['K_Means_Prediction']=self.data['K_Means_Prediction'].replace(0,-1)
        return data

    def isolation_forest(self): 
        from sklearn.ensemble import IsolationForest
        total_predictons = []
        for i in range(0,self.iterations):
            current_prediction = IsolationForest(random_state=0).fit_predict(self.data)
            total_predictons.append(pd.DataFrame(current_prediction).T)
        concatenated = pd.concat(total_predictons)
        round_predictions = round(concatenated.mean(axis=0))
        self.data["Isolation_Forest_Prediction"]=round_predictions
        return data
    
class DIMENSIONALITY_REDUCTION():
    def __init__(self,No_of_components):
        self.components = No_of_components
        self.data = data.copy()
    
    def GET_pca(self):
        pca = PCA(n_components=self.components)
        pca.fit(data)
        print("PCA explains ",round(sum(pca.explained_variance_ratio_),3),"% of the variance in the data")
        data = pca.fit_transform(self.data)
        data = pd.DataFrame(data)
        if self.components == 2:
            data.columns = ["PCA1","PCA2"]
        if self.components == 3:
            data.columns = ["PCA1","PCA2","PCA3"]
        return data
    
    def GET_tsne(self,perplexity=30):
        from sklearn.manifold import TSNE
        n_components=2 
        data =  TSNE(n_components=self.components, 
                     learning_rate="auto",
                     init='random', 
                     perplexity=perplexity).fit_transform(self.data)
        tsne_df = pd.DataFrame(data)
        tsne_df.columns = ["tsne_1","tsne_2"]
        return tsne_df 
    
    def GET_pearson_correlation(self,data):
        data = data.dropna()
        columns = list(data.columns)
        from scipy.stats import pearsonr
        result = []
        for i in range(0,len(columns)):
            for j in range(0,len(columns)):
                if i != j and i > j:
                    col1,col2 = columns[i],columns[j]
                    data1 = data[col1]
                    data2 = data[col2]
                    corr, _ = pearsonr(data1, data2)
                    result.append([col1,col2,corr])
                    #print('Pearsons correlation: %.3f' % corr)
        result = pd.DataFrame(result)
        result.columns = ["Var1","Var2","Correlation"]
        return result
    
    def SET_Variable_Selection(self,X,Y,num_of_top_variables):    
        xgb = XGBRegressor(n_estimators=100)
        xgb.fit(X,Y)
        importance = pd.DataFrame([item*100 for item in xgb.feature_importances_])
        importance["Variable"]=[item for item in X.columns]
        importance.columns = ["Importance","Var_Name"]

        importance = importance.sort_values(by=["Importance"],ascending = False)
        #filtering_out_unncessary_variables = importance[(importance["Importance"] >= 0.1) & (importance["Importance"] <= 1) ]
        #importance = importance[importance["Var_Name"] != "mile_per_kwh"]
        filtering_out_unncessary_variables = importance[0:num_of_top_variables]
        filtered_data = X[[item for item in filtering_out_unncessary_variables["Var_Name"]] ]
        return filtered_data
    
    
def convert_into_data_format(X_train,X_test):
    X_train = X_train.values
    X_test = X_test.values
    train_X = X_train.reshape((X_train.shape[0],1, X_train.shape[1]))
    test_X = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    return train_X, test_X

def extract_scaling(data):
    feature_extraction_class = Feature_Extraction()
    scaled_data, scaling_function = feature_extraction_class.GET_Scaled_data(data)
    return scaled_data, scaling_function

def get_result(X_train, X_test, Y_scale_fx, y_train, y_test):
    xgb_performance = []
    xgb = XGBRegressor(n_jobs=5, learning_rate =0.1, subsample=0.8,
                         max_depth = 5, min_child_weight = 1, gamma = 0, scale_pos_weight = 1)
    model = xgb.fit(X_train,y_train)
    pred = model.predict(X_test)
    pred = Y_scale_fx.inverse_transform([pred])[0]
    xgb_performance.append(mean_absolute_percentage_error(pred,Y_scale_fx.inverse_transform(y_test)))
    return np.mean(xgb_performance)

def sanity_checks(data):
    data = data[data["starting_SOC"] >= 0]
    data = data[data["miles_driven"] > 0]
    data = data[data["mile_per_kwh"] >= 0]
    data = data[data["average_temp"] <= 100]
    data = data[data["kwh_per_mile"] <= 4.5]
    data = data.reset_index().drop(columns = ["index"])
    return data

def Data_Cleaning(data, reduction,max_variables, reduction_type,drop_null_values, 
                  target_variable, 
                  extra_time_features,
                 num_of_components):
    print("The dataset contains ",data.shape[0]," rows and ",data.shape[1]," columns.")
    original_data = data.copy()
    variables_to_remove = ["energy","relative_error",
                        "end_time",
                        "max_speed",
                        "Anomalous",
                        "kwh_per_mile",
                        'energy_consuming',
                        'energy_regenerating',
                        "average_speed_driven",
                        "accerlation_max",
                        "accerlation_avg",
                        "accerlation_min",
                        "calculated_rm_mis",
                        "mile_per_kwh","route",
                        "predicted_rm_mis",
                        "accuracy_loss",
                        "average_humidity",
                        "estimated",
                        "route",
                        "route_id",
                        "trip_id",
                        "day_type",
                        "year",
                        "mile_per_soc",
                        "ending_SOC",
                        "route_id"]

    variables_to_remove = [variable for variable in variables_to_remove if variable != target_variable]
    #print("\n")
    #print("The following variables have been removed", variables_to_remove)
    #print("\n")
    feature_extraction_class = Feature_Extraction()
    data  = feature_extraction_class.GET_deleted_columns(data,variables_to_remove)
    print("\n")
    print("The extra unncessary columns have been deleted")
    print("The dataset now contains ",data.shape[0]," rows and ",data.shape[1]," columns.")
    if extra_time_features == True:
        data = feature_extraction_class.GET_reduce_time_features(data,"start_time")
        print("\n")
        print("The time related features extracted for the start of the trip")
        print("The dataset now contains ",data.shape[0]," rows and ",data.shape[1]," columns.")
        
    if extra_time_features == False:
        print("The time related features have not been extracted")
    
    if drop_null_values == True:
        data = data.dropna().reset_index(inplace = False).drop(columns = ["index","start_time_year","start_time_month"])
        print("\n")
        print("The dataset now contains ",data.shape[0]," rows and ",data.shape[1]," columns.")
        print("The Null Values have been dropped and index resest with extra column deleted")

        
    if drop_null_values == False:
        reduction = False
        null_columns = data.columns[data.isnull().any()].tolist()
        print("Null Values are present in",null_columns)
        print("Null Values have not been dropped")
        
        
    data = feature_extraction_class.GET_BUS_DESIGN(data)
    X_scaled_data,Y_scaled_data = feature_extraction_class.GET_X_and_Y(data,target_variable)
    
    X_data, scaling_function_x = extract_scaling(X_scaled_data)
    Y_data, scaling_function_y = extract_scaling(Y_scaled_data)
    
    print("\n")
    print("The data has been scaled and bus design related features added.")
    print("The dataset now contains ",data.shape[0]," rows and ",data.shape[1]," columns.")
    
    if reduction == True:
        dimenstionality_reduction_class = DIMENSIONALITY_REDUCTION(No_of_components=3)
        X_scaled_data,Y_scaled_data = feature_extraction_class.GET_X_and_Y(data,target_variable)
        X_scaled_data = dimenstionality_reduction_class.SET_Variable_Selection(X=X_scaled_data,
                                                                               Y=Y_scaled_data,
                                                                               num_of_top_variables=max_variables)

        if reduction_type == "pca":
            data = dimenstionality_reduction_class.GET_pca(No_of_components=dimensionality_reduction_dimensions,data=data)
        if reduction_type == "tsne":
            data = dimenstionality_reduction_class.GET_tsne(data)  
        return X_scaled_data, Y_scaled_data, X_scale_fx, Y_scale_fx
    else:
        print("\n")
        print("Dimensionality Reduction has been switched off")
        print("The data has been returned")
        print("\n")
        print("The datset has lost",original_data.shape[0]-data.shape[0]," number of rows")
        return X_data, scaling_function_x,Y_data, scaling_function_y
    
def plot_creations():
    variable = "BATTERY_CAPACITY"
    feature_extraction_class = Feature_Extraction()
    original_data = data.copy()
    X,Y = feature_extraction_class.GET_X_and_Y(data,target_variable)
    X_scaled_data, X_scale_fx = feature_extraction_class.GET_Scaled_data(X)
    Y_scaled_data, Y_scale_fx = feature_extraction_class.GET_Scaled_data(Y)
    X_scaled_data[target_variable] = Y_scaled_data
    data = X_scaled_data
    manu_options = list(set(data[variable]))
    mapping = {}
    for i in range(0,len(data)):
        mapping[X_scaled_data[variable][i]] = str(original_data[variable][i])

    results = []
    feature_extraction_class = Feature_Extraction()
    for i in range(0,len(manu_options)):
        training_data = data[data[variable]==manu_options[i]]
        testing_data = data[data[variable]!=manu_options[i]]
        X_train,Y_train = feature_extraction_class.GET_X_and_Y(training_data,target_variable)
        X_test,Y_test = feature_extraction_class.GET_X_and_Y(testing_data,target_variable)
        results.append([mapping[manu_options[i]],get_result(X_train, X_test,Y_scale_fx, Y_train, Y_test), X_test.shape])
        
data = pd.read_csv("/Users/aasimwani/Downloads/data/culver_city.csv")
drop_null_values = True
dimensionality_reduction_dimensions = 3
max_variables = 10
reduction = False # Fix PCA and TSNE
extra_time_features = True
reduction_type = "pca" #or "tsne"
target_variable = "kwh_per_mile"
X_data, scaling_function_x,Y_data, scaling_function_y = Data_Cleaning(data,reduction, max_variables, reduction_type, drop_null_values, target_variable, extra_time_features,
                             num_of_components = dimensionality_reduction_dimensions)
