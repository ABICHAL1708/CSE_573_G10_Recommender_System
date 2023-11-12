def preprocess_data(data_dir):
	# EDIT
	data_dir = base_path+"/"+data_dir
	print("===========================================")
	print("Preprocessing the Data")
	print("===========================================")
	print("The files present in the data directory are given as-")
	print(os.listdir(data_dir))

	movie_columns = ['movie_id', 'title', 'genres']
	movies = pd.read_table(data_dir+"/movies.dat", sep = "::", header = None, names = movie_columns, encoding = "latin-1")

	rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
	ratings = pd.read_table(data_dir+"/ratings.dat", sep = "::", header = None, names = rating_columns, encoding = "latin-1")

	user_columns = ['user_id', 'gender', 'age', 'occupation', 'zip']
	users = pd.read_table(data_dir+"/users.dat", sep = "::", header = None, names = user_columns, encoding = "latin-1")

	# Preprocessing the ratings to create 
	print(ratings)

	# Creating the dataset object
	dataset = Dataset()

	
	return " "