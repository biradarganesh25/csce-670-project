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

def get_legacy_session():
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
    session = requests.session()
    session.verify = "./crts"
    session.mount('https://', CustomHttpAdapter(ctx))
    return session

def extract_text_from(link_file):
    with open(link_file, 'r') as f:
        urls = f.readlines()
    for i, url in enumerate(urls):
        url = url.strip()
        html = get_legacy_session().get(url).text
        soup = BeautifulSoup(html, features="html.parser")
        text = soup.get_text()

        lines = (line.strip() for line in text.splitlines())
        with open (f'{i}.txt', 'w') as f:
            f.write('\n'.join(line for line in lines if line))
    # return '\n'.join(line for line in lines if line)

#write a function that accepts a url and returns a list of all urls under that url
def get_all_links(url, level):
    print("current url: ************", url)
    if level == 2:
        return []
    html = get_legacy_session().get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    links = []
    for link in soup.find_all('a'):
        actual_link = link.get('href')
        #identify if the link is a relative link
        if actual_link and actual_link[0] == '/':
            print(actual_link)
            actual_link = url + actual_link[1:]            
            links.append(actual_link)
            # links.extend(get_all_links(actual_link, level+1))
    return set(links)

def write_links_to_file():
    links_to_scrape = get_all_links('https://iss.tamu.edu/', 0)
    with open('links.txt', 'w') as f:
        for link in links_to_scrape:
            f.write(link+'\n')

# write_links_to_file()
extract_text_from('links.txt')