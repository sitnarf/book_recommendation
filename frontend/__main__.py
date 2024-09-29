import streamlit as st
import requests


#
# Function to search books by title
def search_books(query):
    response = requests.get(f"http://your_rest_api_url/search?query={query}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error fetching data")
        return []


# Function to get book recommendations
def get_recommendations(book_id):
    response = requests.get(
        f"http://your_rest_api_url/recommendations?book_id={book_id}"
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error fetching data")
        return []


# # Streamlit UI
#
# # Title
st.title("Book Recommendation System")
#
# Search Input
query = st.text_input("Search for a book")

if query:
    # Fetch search results
    books = search_books(query)

    if books:
        st.write("Search Results:")

        for book in books:
            if st.button(book["title"]):
                recommendations = get_recommendations(book["id"])

                if recommendations:
                    st.write("Recommendations:")
                    for rec in recommendations:
                        st.write(rec["title"])
                else:
                    st.write("No recommendations available")
    else:
        st.write("No books found")
