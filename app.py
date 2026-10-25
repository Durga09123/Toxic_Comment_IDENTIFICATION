from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import re
import os

# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)
app.secret_key = "toxic-comment-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model_path = os.path.join(
    BASE_DIR,
    "toxic_model.pkl"
)

tfidf_path = os.path.join(
    BASE_DIR,
    "tfidf_vectorizer.pkl"
)

model = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)


# ============================================================
# MOVIE DETAILS
# ============================================================

movies = [

   {
        "id": 1,
        "name": "Dragon",
        "image": "dragon.jpg",
        "description":
            "Dragon follows Raghuvan, a carefree young man whose college life and relationships take "
            "unexpected turns because of his choices. After facing failures and heartbreak, he decides "
            "to change his life and become successful. His journey takes him through love, friendship, "
            "ambition and the consequences of his decisions."
    },

    {
        "id": 2,
        "name": "Vidaamuyarchi",
        "image": "vidaamuyarchi.jpg",
        "description":
            "Arjun travels to Azerbaijan with his wife Kayal, but their trip takes a dangerous turn when "
            "she suddenly disappears. Arjun refuses to give up and begins searching for her through an "
            "unfamiliar and dangerous environment. His search gradually reveals unexpected people, secrets "
            "and threats that make his mission increasingly difficult."
    },

    {
        "id": 3,
        "name": "Thug Life",
        "image": "thug-life.jpg",
        "description":
            "Thug Life follows Sakthivel, a powerful gangster who rescues a young boy and raises him as "
            "his own son. Their relationship changes as the boy grows older and becomes involved in the "
            "world of crime and power. Loyalty, betrayal, ambition and revenge gradually bring them into "
            "a dangerous conflict."
    },

    {
        "id": 4,
        "name": "Coolie",
        "image": "coolie.jpg",
        "description":
            "Deva is a former gold smuggler who lives away from his dangerous past until circumstances "
            "force him back into the criminal world. He discovers a network connected to valuable vintage "
            "watches and decides to take matters into his own hands. His actions bring him face-to-face "
            "with powerful criminals and create a series of dangerous confrontations."
    },

    {
        "id": 5,
        "name": "Retro",
        "image": "retro.jpg",
        "description":
            "A violent man decides to leave his criminal past behind and start a peaceful life with the "
            "woman he loves. He tries to escape the world of violence and build a future away from his "
            "old enemies. However, the past he left behind refuses to disappear and eventually threatens "
            "his family, relationship and dreams of a peaceful life."
    },

    {
        "id": 6,
        "name": "Tourist Family",
        "image": "tourist-family.jpg",
        "description":
            "A Sri Lankan Tamil family leaves their homeland and secretly travels to Tamil Nadu in search "
            "of a safer and better life. They settle into a new community and slowly develop relationships "
            "with the people around them. Their everyday struggles lead to several humorous and emotional "
            "situations while showing the importance of family, friendship and humanity."
    },

    {
        "id": 7,
        "name": "Madharaasi",
        "image": "madharaasi.jpg",
        "description":
            "A troubled man becomes connected to a dangerous criminal conspiracy after a series of "
            "unexpected events. As he investigates what is happening around him, he discovers that powerful "
            "people are involved. He must overcome his fears and use his determination to protect himself "
            "and confront the people responsible for the danger."
    },

    {
        "id": 8,
        "name": "Ace",
        "image": "ace.jpg",
        "description":
            "A man travels to Malaysia hoping to leave his troubled past behind and start a completely new "
            "life. However, he soon becomes involved in a risky situation involving crime, money and a "
            "planned robbery. His simple plans become complicated when unexpected relationships and dangerous "
            "criminals enter his life."
    },

    {
        "id": 9,
        "name": "Revolver Rita",
        "image": "revolver-rita.jpg",
        "description":
            "Rita finds herself caught in a dangerous situation involving criminals and people searching "
            "for something valuable. Instead of running away, she decides to face the people threatening "
            "her. With courage and intelligence, she tries to survive the dangerous circumstances while "
            "uncovering the truth behind the conflict."
    },

    {
        "id": 10,
        "name": "Veera Dheera Sooran",
        "image": "veera-dheera-sooran.jpg",
        "description":
            "Kaali appears to be an ordinary grocery-store owner who lives peacefully with his family. "
            "Unknown to the people around him, he has a violent and dangerous past that he has carefully "
            "hidden. When his family becomes involved in a dangerous situation, Kaali is forced to return "
            "to the world of violence that he once escaped."
    },

    {
        "id": 11,
        "name": "Kaantha",
        "image": "kaantha.jpg",
        "description":
            "Set during the 1950s, Kaantha takes place in the early period of the Tamil film industry. "
            "The story follows characters whose lives become connected through cinema, ambition and personal "
            "relationships. As their dreams of success grow, they face conflicts involving love, fame, "
            "power and betrayal."
    },

    {
        "id": 12,
        "name": "Jana Nayagan",
        "image": "jana-nayagan.jpg",
        "description":
            "Jana Nayagan follows an influential man who becomes involved in political and social issues "
            "affecting ordinary people. His decision to stand up against powerful forces creates several "
            "challenges in his life. The story focuses on his journey, his relationships and his struggle "
            "to make a difference while facing political opposition."
    },

    {
        "id": 13,
        "name": "Love Insurance Kompany",
        "image": "love-insurance-kompany.jpg",
        "description":
            "The story takes place in a world where technology can be used to manage and protect romantic "
            "relationships. A young man working in this unusual industry becomes personally involved in a "
            "love story. His professional responsibilities and personal feelings begin to clash, creating "
            "a mixture of romance, comedy, technology and unexpected situations."
    },

    {
        "id": 14,
        "name": "Karuppu",
        "image": "karuppu.jpg",
        "description":
            "Karuppu follows a man who becomes involved in a serious conflict affecting his family and "
            "the people around him. As powerful opponents enter the situation, he is forced to confront "
            "difficult choices and protect the people he cares about. The story combines action, family "
            "emotions and a struggle against powerful forces."
    },

    {
        "id": 15,
        "name": "Sardar 2",
        "image": "sardar-2.jpg",
        "description":
            "Sardar 2 continues the spy-action story surrounding the Sardar character and his son Vijay. "
            "Vijay becomes involved in a dangerous investigation connected to an international criminal "
            "threat. As the investigation develops, secrets from the past and a powerful enemy known as "
            "Black Dagger create a much larger conflict."
    },

    {
        "id": 22,
        "name": "29",
        "image": "29.jpg",
        "description":
            "29 is a romantic drama that focuses on relationships, emotions and the experiences of its "
            "central characters. The story follows Sathya and Viji as their lives become connected through "
            "unexpected circumstances. As their relationship develops, they face emotional challenges and "
            "difficult choices that test their understanding of love and life."
    },

    {
        "id": 17,
        "name": "Mandaadi",
        "image": "mandaadi.jpg",
        "description":
            "Mandaadi is set against a coastal backdrop and explores the lives of people connected to "
            "traditional boat racing. The protagonist faces personal struggles while trying to achieve his "
            "dreams and prove himself. The story combines competition, family relationships, determination "
            "and the unique culture of a coastal community."
    },

    {
        "id": 23,
        "name": "Parimala and Co",
        "image": "parimala-and-co.jpg",
        "description":
            "Parimala and Co follows an ordinary family whose peaceful life turns chaotic after they become "
            "connected to a murder. Instead of immediately revealing what happened, the family tries to deal "
            "with the situation while avoiding suspicion. As the police investigation moves closer to them, "
            "their attempts to maintain a normal life create a mixture of comedy, mystery and tension."
    },

    {
        "id": 19,
        "name": "Mr. X",
        "image": "mr-x.jpg",
        "description":
            "Mr. X follows a mysterious character who becomes involved in a dangerous mission surrounded "
            "by secrets and deception. As he moves deeper into the mission, he encounters powerful enemies "
            "and unexpected obstacles. The story combines action, espionage and suspense as the protagonist "
            "tries to uncover the truth."
    },

    {
        "id": 20,
        "name": "Kara",
        "image": "kara.jpg",
        "description":
            "Kara follows a character dealing with complicated relationships, emotional struggles and "
            "difficult circumstances. Unexpected events force the characters to reconsider their choices "
            "and relationships with one another. The story explores love, personal conflicts, emotions and "
            "the search for a better future."
    },

    {
        "id": 21,
        "name": "Vishwanath and Sons",
        "image": "vishwanath-and-sons.jpg",
        "description":
            "Vishwanath and Sons is a family drama centered around a father and his sons who struggle with "
            "different expectations and ambitions. Conflicts over family responsibilities create tension "
            "between the generations. As the story progresses, they must understand each other's feelings "
            "and discover the importance of family relationships."
    }
]


# ============================================================
# STORE ACCEPTED COMMENTS
# ============================================================

movie_comments = {}

for movie in movies:
    movie_comments[movie["id"]] = []


# ============================================================
# STORE BLOCKED COMMENT COUNT
# ============================================================

blocked_counts = {}

for movie in movies:
    blocked_counts[movie["id"]] = 0


# ============================================================
# VULGAR / PROHIBITED WORDS
# ============================================================

BLOCKED_WORDS = {
    "fuck",
    "fucking",
    "fucked",
    "fucker",
    "shit",
    "shitty",
    "bullshit",
    "bitch",
    "asshole",
    "bastard",
    "dick",
    "dickhead",
    "slut",
    "whore",
    "prick",
    "twat",
    "wanker",
    "dipshit",
    "shithead"
}


# ============================================================
# MOVIE-SPECIFIC TOXIC WORDS / PHRASES
# ============================================================

TOXIC_MOVIE_PHRASES = {
    "idiot",
    "idiots",
    "useless",
    "pathetic",
    "horrible",
    "incompetent",
    "annoying",
    "ashamed",
    "no talent",
    "terrible actor",
    "terrible actress",
    "awful actor",
    "awful actress",
    "bad actor",
    "bad actress",
    "worthless",
    "embarrassing",
    "ruined everything",
    "should never act again"
}


# ============================================================
# CHECK VULGAR WORDS
# ============================================================

def contains_blocked_word(text):

    text = text.lower()

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text
    )

    for word in words:

        if word in BLOCKED_WORDS:
            return True

    return False


# ============================================================
# CHECK MOVIE-SPECIFIC TOXIC PHRASES
# ============================================================

def contains_toxic_phrase(text):

    text = text.lower()

    for phrase in TOXIC_MOVIE_PHRASES:

        if phrase in text:
            return True

    return False


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# TOXIC COMMENT PREDICTION
# ============================================================

def predict_comment(comment):

    # First check movie-specific toxic phrases
    if contains_toxic_phrase(comment):

        return "Toxic", 100.0


    # Clean comment
    cleaned_comment = clean_text(comment)


    # Convert comment into TF-IDF
    vector = tfidf.transform(
        [cleaned_comment]
    )


    # Get toxic probability
    probability = model.predict_proba(
        vector
    )[0][1]


    # Toxicity threshold
    threshold = 0.70


    # Classify comment
    if probability >= threshold:

        prediction = "Toxic"

        confidence = probability * 100

    else:

        prediction = "Non-Toxic"

        confidence = (1 - probability) * 100


    return prediction, confidence


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        movies=movies
    )


# ============================================================
# MOVIE DETAILS PAGE
# ============================================================

@app.route("/movie/<int:movie_id>")
def movie(movie_id):

    selected_movie = None

    for item in movies:

        if item["id"] == movie_id:

            selected_movie = item
            break


    if selected_movie is None:

        return "Movie not found", 404


    # Get accepted comments
    comments = movie_comments[movie_id]


    # Count toxic comments
    toxic_count = sum(
        1
        for comment in comments
        if comment["prediction"] == "Toxic"
    )


    # Count non-toxic comments
    non_toxic_count = sum(
        1
        for comment in comments
        if comment["prediction"] == "Non-Toxic"
    )


    # Get blocked comments
    blocked_count = blocked_counts[movie_id]


    # Total comment attempts
    total_comments = (
        len(comments)
        + blocked_count
    )


    # Calculate percentages
    if total_comments > 0:

        toxic_percentage = round(
            (toxic_count / total_comments) * 100,
            1
        )

        non_toxic_percentage = round(
            (non_toxic_count / total_comments) * 100,
            1
        )

        blocked_percentage = round(
            (blocked_count / total_comments) * 100,
            1
        )

    else:

        toxic_percentage = 0
        non_toxic_percentage = 0
        blocked_percentage = 0


    return render_template(
        "movie.html",
        movie=selected_movie,
        comments=comments,
        toxic_count=toxic_count,
        non_toxic_count=non_toxic_count,
        blocked_count=blocked_count,
        total_comments=total_comments,
        toxic_percentage=toxic_percentage,
        non_toxic_percentage=non_toxic_percentage,
        blocked_percentage=blocked_percentage
    )


# ============================================================
# ADD COMMENT
# ============================================================

@app.route(
    "/comment/<int:movie_id>",
    methods=["POST"]
)
def add_comment(movie_id):

    # Get username
    username = request.form.get(
        "username",
        "Anonymous"
    ).strip()


    # Get comment
    comment_text = request.form.get(
        "comment",
        ""
    ).strip()


    # Default username
    if not username:

        username = "Anonymous"


    # Empty comment
    if not comment_text:

        flash(
            "Please enter a comment.",
            "warning"
        )

        return redirect(
            url_for(
                "movie",
                movie_id=movie_id
            )
        )


    # ========================================================
    # FIRST CHECK: VULGAR / PROHIBITED LANGUAGE
    # ========================================================

    if contains_blocked_word(comment_text):

        # Increase blocked count
        blocked_counts[movie_id] += 1


        # Show alert
        flash(
            "⚠ Your comment contains prohibited "
            "language and has been blocked.",
            "blocked"
        )


        # DO NOT SAVE THE COMMENT
        return redirect(
            url_for(
                "movie",
                movie_id=movie_id
            )
        )


    # ========================================================
    # SECOND CHECK: TOXICITY
    # ========================================================

    prediction, confidence = predict_comment(
        comment_text
    )


    # ========================================================
    # CREATE ACCEPTED COMMENT
    # ========================================================

    new_comment = {

        "username": username,

        "text": comment_text,

        "prediction": prediction,

        "confidence": round(
            confidence,
            2
        )
    }


    # ========================================================
    # STORE COMMENT
    # ========================================================

    if movie_id in movie_comments:

        movie_comments[movie_id].append(
            new_comment
        )


    # ========================================================
    # RETURN TO MOVIE PAGE
    # ========================================================

    return redirect(
        url_for(
            "movie",
            movie_id=movie_id
        )
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )