# Standard book genres based on international library classifications
STANDARD_GENRES = [
    # Fiction
    "Literary Fiction",
    "Historical Fiction",
    "Science Fiction",
    "Fantasy",
    "Mystery",
    "Thriller",
    "Horror",
    "Romance",
    "Contemporary Fiction",
    "Crime Fiction",
    "Adventure",
    "Young Adult",
    "Children's Literature",
    "Graphic Novel",
    "Short Stories",
    "Poetry",
    "Drama",
    
    # Non-Fiction
    "Biography",
    "Autobiography",
    "Memoir",
    "History",
    "Philosophy",
    "Religion",
    "Psychology",
    "Science",
    "Technology",
    "Mathematics",
    "Social Sciences",
    "Political Science",
    "Economics",
    "Business",
    "Self-Help",
    "Travel",
    "True Crime",
    "Art",
    "Music",
    "Photography",
    "Cooking",
    "Health",
    "Nature",
    "Reference",
    "Education",
    "Language",
    "Sports",
    "Essays",
    "Journalism"
]

# Genre groupings for better organization
GENRE_CATEGORIES = {
    "Fiction": [
        "Literary Fiction",
        "Historical Fiction",
        "Science Fiction",
        "Fantasy",
        "Mystery",
        "Thriller",
        "Horror",
        "Romance",
        "Contemporary Fiction",
        "Crime Fiction",
        "Adventure",
        "Young Adult",
        "Children's Literature",
        "Graphic Novel",
        "Short Stories",
        "Poetry",
        "Drama"
    ],
    "Non-Fiction": [
        "Biography",
        "Autobiography",
        "Memoir",
        "History",
        "Philosophy",
        "Religion",
        "Psychology",
        "Science",
        "Technology",
        "Mathematics",
        "Social Sciences",
        "Political Science",
        "Economics",
        "Business",
        "Self-Help",
        "Travel",
        "True Crime",
        "Art",
        "Music",
        "Photography",
        "Cooking",
        "Health",
        "Nature",
        "Reference",
        "Education",
        "Language",
        "Sports",
        "Essays",
        "Journalism"
    ]
}

# Genre prompt for the LLM
GENRE_PROMPT = """
Please select the most appropriate genres for this book from the following list. Choose up to 3 genres that best describe the work:

{genres}

Format your response as a JSON array containing only genres from this list. Example:
["Literary Fiction", "Historical Fiction"]
""" 