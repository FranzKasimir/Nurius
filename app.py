from flask import Flask, render_template, request, redirect, url_for, session
from googleapiclient.discovery import build
from textblob import TextBlob
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import nltk
import re
import os

nltk.download('punkt')

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_session_protection'

USERNAME = 'Nurius'
PASSWORD = 'Curius'

YOUTUBE_API_KEY = "AIzaSyARqmLKUxE8TRZHeI2K8MXMWhehKWsF8VQ"
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('username')
        pw = request.form.get('password')
        if user == USERNAME and pw == PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if not session.get('authenticated'):
        return redirect(url_for('login'))

    summary = ""
    sentiments = {}
    keywords = []
    comments = []

    if request.method == "POST":
        video_url = request.form.get("youtube_url")
        video_id = extract_video_id(video_url)

        if video_id:
            comments = get_comments(video_id)
            keywords = extract_keywords(comments)
            sentiments = analyze_sentiment(comments)
            summary = summarize_comments(comments)

    return render_template("index.html", summary=summary, sentiments=sentiments, keywords=keywords)

def extract_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def get_comments(video_id):
    comments = []
    next_page_token = None
    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        ).execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        next_page_token = response.get("nextPageToken")
        if not next_page_token or len(comments) > 500:
            break
    return comments

def extract_keywords(comments):
    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    X = vectorizer.fit_transform(comments)
    return vectorizer.get_feature_names_out()

def analyze_sentiment(comments):
    result = {"positiv": 0, "negativ": 0, "neutral": 0}
    for comment in comments:
        blob = TextBlob(comment)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            result["positiv"] += 1
        elif polarity < -0.1:
            result["negativ"] += 1
        else:
            result["neutral"] += 1
    return result

def summarize_comments(comments):
    if not comments:
        return "Keine Kommentare gefunden."
    joined = " ".join(comments[:50])
    result = summarizer(joined, max_length=130, min_length=30, do_sample=False)
    return result[0]['summary_text']

if __name__ == "__main__":
    app.run(debug=True)
