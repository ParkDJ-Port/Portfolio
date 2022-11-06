## ABAE + Seed_words(ARYA)

# python train.py --emb-name ../preprocessed_data/restaurant/w2v_embedding --embdim 200 --aspect-size 5 --domain restaurant --seed-word ../preprocessed_data/restaurant/seed_words.txt

# python evaluation.py --aspect-size 5 --seed-word ../preprocessed_data/restaurant/seed_words.txt 


## ABAE + no seed words
# python train.py --emb-name ../preprocessed_data/restaurant/w2v_embedding --embdim 200 --aspect-size 5 --domain restaurant

# python evaluation.py --aspect-size 5


## ABAE + Seed_words(retrieved from no seed ABAE log)
# python train.py --emb-name ../preprocessed_data/restaurant/w2v_embedding --embdim 200 --aspect-size 5 --domain restaurant --seed-word ../preprocessed_data/restaurant/seed_words_init.txt

python evaluation.py --aspect-size 5 --seed-word ../preprocessed_data/restaurant/seed_words_init.txt 