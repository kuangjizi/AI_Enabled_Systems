import pandas as pd
import numpy as np


class RuleBasedFiltering:
    """
    A rule-based filtering recommender system.

    This class uses item metedata to recommend top-rated items overall or by genre.
    """

    def __init__(self, ratings_file, item_metadata_file, user_metadata_file=None):
        """
        Initialize the recommender system and load data.

        Parameters
        ----------
        ratings_file : str
            Path to the ratings dataset file.
        metadata_file : str
            Path to the item metadata file.
        """
        self.ratings_file = ratings_file
        self.item_metadata_file = item_metadata_file
        self.user_metadata_file = user_metadata_file
        self.ratings = None
        self.items_metadata = None
        self.user_metadata = None
        self._load_data()

    def _load_data(self):
        """Load the source data"""

        # Ratings data
        self.ratings = pd.read_csv(
            self.ratings_file,
            sep='\t',
            names=['user', 'item', 'rating', 'timestamp']
        )
       
        # Item metadata
        self.items_metadata = pd.read_csv(
            self.item_metadata_file,
            sep='|',
            encoding='latin-1',
            names=[
                'item', 'title', 'release_date', 'video_release_date',
                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
            ]
        )
        # Combine genre columns into a single 'features' column
        self.items_metadata['features'] = self.items_metadata.iloc[:, 6:].apply(
            lambda x: ' '.join([col for col in self.items_metadata.columns[6:] if x[col] == 1]),
            axis=1
        )

        # User metadata (optional)
        if self.user_metadata_file:
            self.user_metadata = pd.read_csv(
                self.user_metadata_file,
                sep='|',
                encoding='latin-1',
                names=[
                    'user', 'age', 'gender', 'occupation', 'zip code'
                ]
            )

    def recommend(self, k=5, rule='overall', criteria=None):
        """
        Recommend top-rated items based on the specified rule.

        Parameters
        ----------
        k : int, optional
            Number of top-rated items to return.
        rule : str, optional
            The rule to apply for recommendations ('overall', 'by_genre',  'by_user_occupation').
        criteria : str, optional
            The criteria to filter items by (e.g., 'Comedy', 'Action', 'artist'
        """
        if rule == 'overall':
            return self._get_top_items(k)
        
        elif rule == 'by_genre':
            if criteria is None:
                raise ValueError("Please specify a genre to filter by.")
            return self._get_top_items_by_genre(criteria, k)
        
        elif rule == "by_gender":
            if criteria is None:
                raise ValueError("Please specify a user gender to filter by.")
            return self._get_top_items_by_user_gender(criteria, k)
        
        elif rule == "by_occupation":
            if criteria is None:
                raise ValueError("Please specify an occupation to filter by.")
            return self._get_top_items_by_user_occupation(criteria, k)
        
        else:
            raise ValueError(f"Unknown rule: {rule}. Supported rules are 'overall', 'by_genre', 'by_gender' and 'by_occupation'.")
        
    def _get_top_items(self, k=5):
        """
        Get the top-rated items based on the overall ratings.

        Parameters
        ----------
        k : int, optional
            Number of top-rated items to return.

        Returns
        -------
        list
            List of top-rated item id and title tuples.
        """
        # rank items by overall ratings and return top n items
        top_items_overall = self.ratings.groupby('item').agg({'rating': 'mean'}).reset_index()
        top_items_overall = top_items_overall.sort_values(by='rating', ascending=False)

        # merge with item metadata to get item details
        top_items_overall = top_items_overall.merge(self.items_metadata[['item', 'title']], on='item', how='left')

        return top_items_overall.head(k)[['item', 'title']].values.tolist()

    def _get_top_items_by_genre(self, genre, k=5):
        """
        Get the top-rated items based on the specified genre.

        Parameters
        ----------
        genre : str
            The genre to filter items by.
        k : int, optional
            Number of top-rated items to return.

        Returns
        -------
        list
            List of top-rated item id and title tuples in the specified genre.
        """
        # filter items by the specified genre 
        mask = self.items_metadata['features'].str.contains(genre, case=False, na=False) # ignore case and NaN
        items_by_genre = self.items_metadata[mask]

        # Merge with ratings to get ratings for items in the specified genre
        items_by_genre_ratings = items_by_genre[['item', 'title']].merge(
            self.ratings[['item', 'rating']], on='item', how='left'
        )
        # rank items by ratings and return top n items
        top_items_by_genre = items_by_genre_ratings.groupby(['item', 'title']).agg({'rating': 'mean'}).reset_index()
        top_items_by_genre = top_items_by_genre.sort_values(by='rating', ascending=False)
        return top_items_by_genre.head(k)[['item', 'title']].values.tolist()


    def _get_top_items_by_user_gender(self, gender, k=5):
        """
        Get the top-rated items based on the specified gender

        Parameters
        ----------  
        gender : str
            The gender to filter user ratings by.
        k : int, optional
            Number of top-rated items to return.

        Returns
        -------
        list
            List of top-rated items (tuple of item_id, item_title) by the specified user gender.
        """
        if self.user_metadata is None:
            raise Exception("User metadata is not available. Please provide user metadata file.")

        # Filter users  
        users_by_gender = self.user_metadata[self.user_metadata['gender'].str.lower() == gender.lower()]

        # Get ratings from users of the specified
        items_by_gender_ratings = users_by_gender.merge( # merge user metadata with ratings
            self.ratings[['user', 'item', 'rating']],
            on='user', how='left').merge(
            self.items_metadata[['item', 'title']], # merge with item metadata to get item details
            on='item', how='left'   
        )

        # rank items by ratings and return top n items
        top_items_by_gender = items_by_gender_ratings.groupby(['item', 'title']).agg({'rating': 'mean'}).reset_index()
        top_items_by_gender = top_items_by_gender.sort_values(by='rating', ascending=False)
        return top_items_by_gender.head(k)[['item', 'title']].values.tolist()
    
    def _get_top_items_by_user_occupation(self, occupation, k=5):
        """
        Get the top-rated items based on the specified occupation

        Parameters
        ----------  
        occupation : str
            The occupation to filter user ratings by.
        k : int, optional
            Number of top-rated items to return.

        Returns
        -------
        list
            List of top-rated items (tuple of item_id, item_title) by the specified user occupation.
        """
        if self.user_metadata is None:
            raise Exception("User metadata is not available. Please provide user metadata file.")

        # Filter users  
        users_by_occu = self.user_metadata[self.user_metadata['occupation'].str.lower() == occupation.lower()]

        # Get ratings from users of the specified
        items_by_occu_ratings = users_by_occu.merge( # merge user metadata with ratings
            self.ratings[['user', 'item', 'rating']],
            on='user', how='left').merge(
            self.items_metadata[['item', 'title']], # merge with item metadata to get item details
            on='item', how='left'   
        )

        # rank items by ratings and return top n items
        top_items_by_occu = items_by_occu_ratings.groupby(['item', 'title']).agg({'rating': 'mean'}).reset_index()
        top_items_by_occu = top_items_by_occu.sort_values(by='rating', ascending=False)
        return top_items_by_occu.head(k)[['item', 'title']].values.tolist()

    def evaluate(self, recommended, validset_df):
        """
        Evaluate the model by calculating the nDCG score on specified validation set.

        Parameters
        ----------
        validset_df: dataframe
            The test dataset to evaluate model performance
        k: int, optional
            The top k items to calculate nDCG@k

        Returns
        -------
        float
            nDCG value.
        """

        # Function to calculate DCG
        def dcg(ratings):
            ratings = np.asfarray(ratings)
            ranks = np.arange(1, len(ratings) + 1) # ranking from 1
            return np.sum(ratings / np.log2(ranks + 1))
        
        if validset_df is None:
            raise Exception ("Please provide validation set.")
        
         # Group validation ratings per user
        user_true_ratings = (
            validset_df.groupby('user')[['item', 'rating']]
            .apply(lambda g: dict(zip(g['item'], g['rating'])))
            .to_dict()
        )

        user_ndcgs = []
        for user, item_ratings in user_true_ratings.items():
            # Get user's rating for recommended items (0 for no ratings)
            true_ratings = [item_ratings.get(item[0], 0) for item in recommended]

            # Get ideal score for k recommended items
            # Note that this is a different setup from the lecture, where the highest ranking score (5) was used.
            # This is because the user can have fewer than k relevant items and using the higest ranking score may overestimate the ideal case.
            # Using the sorted rating reflect the following question:
            # "What would DCG be if we perfectly ordered the recommended items by their true relevance?"
            ideal_ratings = sorted(true_ratings, reverse=True)
            
            # Calcualte nDCG
            dcg_val = dcg(true_ratings)
            idcg_val = dcg(ideal_ratings)
            ndcg = dcg_val / idcg_val if idcg_val > 0 else 0.0
            user_ndcgs.append(ndcg)
        
        return np.mean(user_ndcgs)


if __name__ == "__main__":
    users = [0, 1, 196]  # Example users
    recommender = RuleBasedFiltering(
        ratings_file='storage/u.data',
        item_metadata_file='storage/u.item',
        user_metadata_file='storage/u.user'  # Optional user metadata file
    )

    # ----- Overall recommendation ----- #
    recommended_overall = recommender.recommend(k=5, rule='overall')  
    for user_id in users:
        print(f"Recommended items for user {user_id}: {"; ".join([item[1] for item in recommended_overall])}")

    # ----- Genre-based recommendation ----- #
    recommended_genre = recommender.recommend(k=5, rule='by_genre', criteria='Comedy')   
    for user_id in users:
        print(f"Recommended Comedy items for user {user_id}: {'; '.join([item[1] for item in recommended_genre])}")
    
    # ----- Evaluation ----- #
    # validset_df = pd.read_csv('storage/test', sep='\t', names=['user', 'item', 'rating', 'timestamp'])
   
    # ndcg_overall = recommender.evaluate(recommended_overall, validset_df)
    # print(f"The average nDCG for top-rating recommendations by all items: {ndcg_overall}")

    # ndcg_genre = recommender.evaluate(recommended_genre, validset_df)
    # print(f"The average nDCG for top-rating recommendations by genre: {ndcg_genre}")