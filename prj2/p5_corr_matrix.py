# Pandas read CSV
sf_train = pd.read_csv('data.csv')

# Correlation Matrix for target
corr_matrix = sf_train.corr()
print(corr_matrix['type'])

# Drop unnecessary columns
sf_train.drop(sf_train.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)
print(sf_train.head())