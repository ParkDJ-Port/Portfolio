config = {
    'domain': 'airbnb',
    'device': 'gpu' #gpu
}
bert_mapper = {
    'airbnb': 'roberta-base'
}
path_mapper = {
    'airbnb': './datasets/tripadvisor'
}
aspect_category_mapper = {
    'airbnb': ['location', 'facilities', 'service', 'mood', 'host']
}
aspect_seed_mapper = {
    'airbnb': {
        'location': {"location", "taxi", "route", "boston", "uber", "Downtown", "st."},
        'facilities': {"facilites", "restaurant", "starbucks", "aquarium", "laundry"},
        'service': {"dinner", "checkin", "airconditioner", "heat", "resolution", "hair dryer", "wifi", "bed", "bathroom", "mattress"},
        'mood': {"mood", "place", "interior", "atmosphere", "picturesque"},
        'host': {"host", "worker", "Anne", "she", "he", "waiter", }
    }
}
sentiment_category_mapper = {
    'airbnb': ['negative', 'positive']
}
sentiment_seed_mapper = {
    'airbnb': {
        'positive': {"good", "great", 'nice', "excellent", "perfect", "impressed", "best", "thin", "cheap", "fast", "clean", "stylish", "lovely", "cute", "awesome", "gracious", "professional", "near"},
        'negative': {"bad", "disappointed", "terrible", "horrible", "small", "broken", "complaint", "junk", "smelled", "uncomfortable", "dirty", "crummy"}
    }
}
M = {
    'airbnb': 150 #100
}
K_1 = 10
K_2 = 30
lambda_threshold = 0.5
batch_size = 32
validation_data_size = 100
learning_rate = 1e-5
epochs = 20
