# Pandas read CSV
sf_train = pd.read_csv('data.csv')

# Correlation Matrix for target
corr_matrix = sf_train.corr()
print(corr_matrix['type'])

# Drop unnecessary columns
sf_train.drop(sf_train.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)
print(sf_train.head())

# Pandas read Validation CSV
sf_val = pd.read_csv('test.csv')

# Drop unnecessary columns
sf_val.drop(sf_val.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)

# Get Pandas array value (Convert to NumPy array)
train_data = sf_train.values
val_data = sf_val.values

# Use columns 2 to last as Input
train_x = train_data[:,2:]
val_x = val_data[:,2:]

# Use columns 1 as Output/Target (One-Hot Encoding)
train_y = to_categorical( train_data[:,1] )
val_y = to_categorical( val_data[:,1] )