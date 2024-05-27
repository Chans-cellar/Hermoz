from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import pandas as pd
import os
from lxml import html


# Initialize the WebDriver
driver = webdriver.Chrome()


# Function to log in to Facebook (You'll need to implement this)
def login_to_facebook(email, password):
    driver.get("https://www.facebook.com")
    time.sleep(2)  # Wait for the page to load
    email_field = driver.find_element("id", "email")
    password_field = driver.find_element("id", "pass")
    email_field.send_keys(email)
    password_field.send_keys(password)
    driver.find_element("name", "login").click()
    time.sleep(5)  # Wait for the login to complete


# Function to scrape comments from a given post URL
def scrape_facebook_post_comments_by_class_name(post_url):
    mobile_url = ''
    if 'https://web.' in post_url:
        mobile_url = post_url.replace('https://web.', 'https://mbasic.')

    elif 'https://www.' in post_url:
        mobile_url = post_url.replace('https://www.', 'https://mbasic.')

    print(mobile_url)

    driver.get(mobile_url)
    time.sleep(15)  # Wait for comments to load

    # initializing scraping through beautifulsoup
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    comments = []

    # Find comments, this is a simplified example, the actual implementation may vary
    for comment in soup.find_all("div", {"class": "_55wr"}):

        commenter_name_object = comment.find("a", {"class": "_1s79 _52jh"})
        if not commenter_name_object is None:
            commenter_name = commenter_name_object.text
        else:
            commenter_name = ''

        comment_text_obj = comment.find("div", {"class": "_14ye"})
        if not comment_text_obj is None:
            comment_text = comment_text_obj.text
        else:
            comment_text = ''

        comments.append((commenter_name, comment_text))

    return comments


def scrape_facebook_post_comments_of_photos_by_xpath(post_url):
    driver.get(post_url)
    time.sleep(15)  # Wait for comments to load

    # Initializing scraping through BeautifulSoup
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    # Convert BeautifulSoup object to lxml etree
    lxml_tree = html.fromstring(str(soup))

    # XPath expression for the parent div containing comments
    parent_div_xpath = "/html/body/div[1]/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]"

    # Use XPath with lxml etree to find the parent div
    parent_divs = lxml_tree.xpath(parent_div_xpath)

    comments = []
    xpath_Index = 1

    # Check if the parent divs are found
    if parent_divs:
        for parent_div in parent_divs:
            # Loop through all the child div elements of the parent div
            child_divs = parent_div.xpath(".//div")  # Select all child div elements
            for child_div in child_divs:
                # --COMMENTATOR NAME--
                commentator_name_object = child_div.xpath(parent_div_xpath + "/div[" + str(
                                                         xpath_Index) + "]/div/div[1]/div[2]/div[2]/div[1]/div[1]/div/div[1]/div/div/span/a/span/span")

                commenter_name = commentator_name_object[0].text if commentator_name_object else ''

                # --COMMENT--
                comment_text_object = child_div.xpath(parent_div_xpath + "/div[" + str(
                                                  xpath_Index) + "]/div/div[1]/div[2]/div[2]/div[1]/div[1]/div/div[1]/div/div/div/span/div/div")
                comment_text = comment_text_object[0].text if comment_text_object else ''

                comments.append((commenter_name, comment_text))
                xpath_Index += 1
    else:
        print("Parent div not found.")

    return comments

def scrape_facebook_post_comments_of_videos_by_xpath(post_url):
    driver.get(post_url)
    time.sleep(15)  # Wait for comments to load

    # Initializing scraping through BeautifulSoup
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    # Convert BeautifulSoup object to lxml etree
    lxml_tree = html.fromstring(str(soup))

    # XPath expression for the parent div containing comments
    parent_div_xpath = "/html/body/div[1]/div/div[1]/div/div[3]/div/div/div[1]/div[2]/div/div/div/div[1]/div/div/div[2]/div[3]/div[1]/div[2]"

    # Use XPath with lxml etree to find the parent div
    parent_divs = lxml_tree.xpath(parent_div_xpath)

    comments = []
    xpath_Index = 1

    # Check if the parent divs are found
    if parent_divs:
        for parent_div in parent_divs:
            # Loop through all the child div elements of the parent div
            child_divs = parent_div.xpath(".//div")  # Select all child div elements
            for child_div in child_divs:
                # --COMMENTATOR NAME--
                commentator_name_object = child_div.xpath(parent_div_xpath + "/div[" + str(
                                                         xpath_Index) + "]/div[8]/div/div[1]/div/div[2]/div[1]/div[1]/div/div/div/div/span/a/span/span")

                commenter_name = commentator_name_object[0].text if commentator_name_object else ''

                # --COMMENT--
                comment_text_object = child_div.xpath(parent_div_xpath + "/div[" + str(
                                                  xpath_Index) + "]/div[8]/div/div[1]/div/div[2]/div[1]/div[1]/div/div/div/div/div/span/div/div")
                comment_text = comment_text_object[0].text if comment_text_object else ''

                comments.append((commenter_name, comment_text))
                xpath_Index += 1
    else:
        print("Parent div not found.")

    return comments


def get_facebook_comments(urlList):
    comment_data_obj = {}
    for url in urlList:
        comments = scrape_facebook_post_comments_by_class_name(url)
        comment_data_obj[url] = comments

    return comment_data_obj


# Example usage
login_to_facebook("economicrerandd@gmail.com", "economicrandd123")

urls = ["https://www.facebook.com/photo/?fbid=849152233881310&set=a.528846962578507"]

comment_data = get_facebook_comments(urls)

print(comment_data)

# Check if the file 'facebook_comments.csv' exists
if os.path.exists('facebook_comments.csv'):
    # Load existing data from the file
    df = pd.read_csv('facebook_comments.csv')
else:
    # Create an empty DataFrame
    df = pd.DataFrame()

# Iterate over each key-value pair in the dictionary
for url, comments in comment_data.items():
    # Convert the comments list of tuples into a DataFrame
    comments_df = pd.DataFrame(comments, columns=['User', 'Comment'])
    # Add a new column to store the URL
    comments_df['URL'] = url
    # Concatenate the comments DataFrame with the main DataFrame
    df = pd.concat([df, comments_df], ignore_index=True)
    print(df)

# Write the DataFrame to 'facebook_comments.csv'
df.to_csv('facebook_comments.csv', index=False)

driver.close()
