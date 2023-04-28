from bs4 import BeautifulSoup
import requests
import urllib3
import ssl

class CustomHttpAdapter (requests.adapters.HTTPAdapter):
    # "Transport adapter" that allows us to use custom ssl_context.

    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = urllib3.poolmanager.PoolManager(
            num_pools=connections, maxsize=maxsize,
            block=block, ssl_context=self.ssl_context)

class DataExtractor:

    def __init__(self,url,links_filename):
        self.parent_website_url = url
        self.links_filename = links_filename


    def get_legacy_session(self):
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        session = requests.session()
        session.verify = "./crts"
        session.mount('https://', CustomHttpAdapter(ctx))
        return session

    #Function to return a list of all urls under a parent url
    def get_all_links_from_url(self, level):
        print("Extracting all urls under {}".format(self.parent_website_url))
        if level == 2:
            return []
        html = self.get_legacy_session().get(self.parent_website_url).text
        soup = BeautifulSoup(html, features="html.parser")

        links = []
        for link in soup.find_all('a'):
            actual_link = link.get('href')
            #identify if the link is a relative link
            if actual_link and actual_link[0] == '/':
                print(actual_link)
                actual_link = self.parent_website_url + actual_link[1:]            
                links.append(actual_link)
        return set(links)

    #Function to write extracted links to a file
    def write_links_to_file(self):
        links_to_scrape = self.get_all_links_from_url(0)
        with open(self.links_filename, 'w') as f:
            for link in links_to_scrape:
                f.write(link+'\n')

    # Extract text from each link and write it to *.txt files. 
    def extract_text_from_links(self):
        with open(self.links_filename, 'r') as f:
            urls = f.readlines()
        for i, url in enumerate(urls):
            url = url.strip()
            html = self.get_legacy_session().get(url).text
            soup = BeautifulSoup(html, features="html.parser")
            text = soup.find_all("main")[0].get_text()
            lines = (line.strip() for line in text.splitlines())
            with open (f'./data/{i}.txt', 'w') as f:
                f.write('\n'.join(line for line in lines if line))

if __name__ == "__main__":

    PARENT_WEBSITE_URL = "https://iss.tamu.edu/"
    LINKS_FILE = "links.txt"

    data_extractor = DataExtractor(PARENT_WEBSITE_URL,LINKS_FILE)
    data_extractor.write_links_to_file()
    data_extractor.extract_text_from_links()